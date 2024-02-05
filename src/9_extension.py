import os
import pandas as pd
import numpy as np
import math
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import plotnine as pn
from sspipe import p, px

# Set up paths
derived_path = os.path.abspath('..') | p(os.path.join, 'data', 'derived')
figure_path = os.path.abspath('..') | p(os.path.join, 'figure')
# Read data
sample = pd.read_csv(os.path.join(derived_path, 'sample_cleaned.csv'),
                     low_memory=False,
                     parse_dates=['retire_month'])  # datetime info will not be preserved in to_csv

# change column type of gor_1 to 'category' (for later use of XGBoost)
sample['gor_1'] = sample['gor_1'].astype('category')

# Assume people are all born in July
sample['birth_month'] = (sample['indobyr_1'].astype(str) + '-6') | p(pd.to_datetime)

# Calculate difference in month
sample['retire_month_diff'] = (
        sample['retire_month'].dt.to_period('M') - sample['birth_month'].dt.to_period('M')).apply(
    lambda x: x.n if pd.notnull(x) else None)
# check
# sample['retire_month_diff'].notnull() | p(sum) # 181 (192 retired respondents)

x_list = ['sex_1', 'married_1', 'grandchild_1', 'degree_1', 'below_degree_1', 'a_levels_1', 'o_levels_1', 'no_qual_1',
          'early_retire_incentive_1', 'pen_db_1', 'pen_dc_1', 'pen_any_1', 'gor_1']

########## The original model (OLS regression on whole sample)
model_origin = smf.ols(formula='retire_month_diff ~ ' + ' + '.join(x_list), data=sample).fit()
model_origin.summary2() | p(print)  # R = 0.294, N = 176, Adjusted R = 0.203 (*)

########## Divide into training and test set
treatment_xy = sample.loc[sample['treatment'] == 1, ['retire_month_diff'] + x_list].dropna()  # N = 176
X_train, X_test, y_train, y_test = train_test_split(treatment_xy.drop('retire_month_diff', axis=1),
                                                    treatment_xy[['retire_month_diff']],
                                                    random_state=300)

########## Apply the original model on the test data
y_pred_origin = model_origin.predict(X_test) + (model_origin.resid | p(np.random.choice,
                                                                     size=len(X_test))) \
                | p(round)

# Take a look at the residual distribution
resid_origin = pd.DataFrame({'resid': model_origin.resid})
resid_fig = (pn.ggplot(resid_origin, pn.aes('resid')) +
             pn.geom_histogram(bins=15, fill='white', colour='black') +
             pn.theme_bw() +
             pn.labs(x='Residual', y='Count') +
             pn.scale_x_continuous(breaks=np.arange(-75, 100, 25)))
pn.ggsave(resid_fig, filename=os.path.join(figure_path, 'resid_origin.png'))

# Calculate some metrics
mae_origin = mean_absolute_error(y_true=y_test, y_pred=y_pred_origin)
rmse_origin = mean_squared_error(y_true=y_test, y_pred=y_pred_origin, squared=False)

########## New approach: XGBoost
D_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
D_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# # CV
# params_cv = {
#     'objective': 'reg:squarederror',
#     'max_depth': 6,
#     'eta': 0.3,
#     'subsample': 1
# }
# num_boost_round = 999
#
# cv_results = xgb.cv(
#     params_cv,
#     D_train,
#     num_boost_round=num_boost_round,
#     nfold=5,
#     metrics={'rmse'},
#     seed=300
# )
#
# print(cv_results)

# Use the optimal parameters
params_xgb = {'objective': 'reg:squarederror'}
n = 100
model_xgb = xgb.train(
    params=params_xgb,
    dtrain=D_train,
    num_boost_round=n,
)

y_pred_xgb = model_xgb.predict(D_test) | p(np.round)

# Calculate some metrics
mae_xgb = mean_absolute_error(y_true=y_test, y_pred=y_pred_xgb)
rmse_xgb = mean_squared_error(y_true=y_test, y_pred=y_pred_xgb, squared=False)

########## Predict on the full sample
sample_xy = sample[x_list + ['idauniq', 'retire_month_diff']].dropna(subset=x_list, axis=0)
D_sample = xgb.DMatrix(sample_xy[x_list], enable_categorical=True)
sample_xy['pred_retire_month_diff'] = model_xgb.predict(D_sample) | p(np.round)

sample = pd.merge(sample, sample_xy[['idauniq', 'pred_retire_month_diff']], how='left', on='idauniq')
# sample['pred_retire_month_diff'].value_counts(dropna=False) # seems about right

sample['pred_retire_month'] = sample.apply(
    lambda row: (pd.DateOffset(months=row['pred_retire_month_diff']) + row['birth_month'])
    if not (pd.isna(row['pred_retire_month_diff']) or pd.isna(row['birth_month']))
    else None,
    axis=1
)

sample['final_retire_month'] = np.where(sample['treatment'] == 0, sample['pred_retire_month'], sample['retire_month'])
# check
# pd.isna(sample['final_retire_month']).sum()
# there are 36 NAs is the final retirement month, as there are missing values in some of the predictors


########## Disease date vs. Retirement date
# no NAs in angina_1, and NAs in angina_2 and angina_3 all mean 'not applicable' rather than e.g. refusal
# angina_1 is binary, angina_2 and angina_3 are datetime
def disease_pre_retire(row, name):
    if pd.isna(row[f'{name}_2']) & pd.isna(row[f'{name}_3']):
        return np.where(row[f'{name}_1'] == 1, 1, 0)
    elif pd.isna(row['final_retire_month']):
        return None
    elif pd.notna(row[f'{name}_2']):
        return np.where(pd.to_datetime(row[f'{name}_2']) <= row['final_retire_month'], 1, 0)
    elif pd.notna(row[f'{name}_3']):
        return np.where(pd.to_datetime(row[f'{name}_3']) <= row['final_retire_month'], 1, 0)


sample['angina_pre'] = sample.apply(disease_pre_retire, name='angina', axis=1)
sample['heart_attack_pre'] = sample.apply(disease_pre_retire, name='heart_attack', axis=1)
sample['stroke_pre'] = sample.apply(disease_pre_retire, name='stroke', axis=1)
sample['diabetes_pre'] = sample.apply(disease_pre_retire, name='diabetes', axis=1)
sample['arthritis_pre'] = sample.apply(disease_pre_retire, name='arthritis', axis=1)
sample['cancer_pre'] = sample.apply(disease_pre_retire, name='cancer', axis=1)
sample['psych_pre'] = sample.apply(disease_pre_retire, name='psych', axis=1)


def disease_post_retire(row, name):
    if row[f'{name}_1'] == 1:
        return 0
    elif pd.isna(row[f'{name}_2']) & pd.isna(row[f'{name}_3']):
        return 0
    elif pd.isna(row['final_retire_month']):
        return None
    elif pd.notna(row[f'{name}_2']):
        return np.where(pd.to_datetime(row[f'{name}_2']) > row['final_retire_month'], 1, 0)
    elif pd.notna(row[f'{name}_3']):
        return np.where(pd.to_datetime(row[f'{name}_3']) > row['final_retire_month'], 1, 0)


sample['angina_post'] = sample.apply(disease_post_retire, name='angina', axis=1)
sample['heart_attack_post'] = sample.apply(disease_post_retire, name='heart_attack', axis=1)
sample['stroke_post'] = sample.apply(disease_post_retire, name='stroke', axis=1)
sample['diabetes_post'] = sample.apply(disease_post_retire, name='diabetes', axis=1)
sample['arthritis_post'] = sample.apply(disease_post_retire, name='arthritis', axis=1)
sample['cancer_post'] = sample.apply(disease_post_retire, name='cancer', axis=1)
sample['psych_post'] = sample.apply(disease_post_retire, name='psych', axis=1)

########## Below is preparation for 7_main_analysis.py
# Any newly diagnosed disease (post)
disease_list = ['angina', 'heart_attack', 'stroke', 'diabetes', 'arthritis', 'cancer', 'psych']
sample['any_post'] = (sample[[disease + '_post' for disease in disease_list]] == 1).any(axis=1).astype(int)
# Any diagnosed disease (pre)
sample['any_pre'] = (sample[[disease + '_pre' for disease in disease_list]] == 1).any(axis=1).astype(int)
# Newly diagnosed angina, heart attack or stroke (post)
sample['angina_heart_attack_diabetes_post'] = (sample[['angina_post', 'heart_attack_post', 'diabetes_post']] == 1).any(
    axis=1).astype(int)


# Any newly diagnosed disease (between w1 and retirement)
def disease_between(row, name):
    if row[f'{name}_1'] == 1:
        return 0
    elif pd.isna(row[f'{name}_pre']):
        return None
    elif row[f'{name}_pre'] == 1:
        return 1
    elif row[f'{name}_pre'] == 0:
        return 0


sample['angina_between'] = sample.apply(disease_between, name='angina', axis=1)
sample['heart_attack_between'] = sample.apply(disease_between, name='heart_attack', axis=1)
sample['stroke_between'] = sample.apply(disease_between, name='stroke', axis=1)
sample['diabetes_between'] = sample.apply(disease_between, name='diabetes', axis=1)
sample['arthritis_between'] = sample.apply(disease_between, name='arthritis', axis=1)
sample['cancer_between'] = sample.apply(disease_between, name='cancer', axis=1)
sample['psych_between'] = sample.apply(disease_between, name='psych', axis=1)

sample['any_between'] = (sample[[disease + '_between' for disease in disease_list]] == 1).any(axis=1).astype(int)

########## Save data
sample.to_csv(os.path.join(derived_path, 'sample_simulate_retire_ext.csv'), index=False)

########## Inspection
