import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import plotnine as pn
from sspipe import p, px

# Set up paths
derived_path = os.path.abspath('..') | p(os.path.join, 'data', 'derived')
# Read data
sample = pd.read_csv(os.path.join(derived_path, 'sample_cleaned.csv'),
                     low_memory=False,
                     parse_dates=['retire_month']) # datetime info will not be preserved in to_csv

# Assume people are all born in July
sample['birth_month'] = (sample['indobyr_1'].astype(str) + '-6') | p(pd.to_datetime)

# calculate difference in month
sample['retire_month_diff'] = (sample['retire_month'].dt.to_period('M') - sample['birth_month'].dt.to_period('M')).apply(lambda x: x.n if pd.notnull(x) else None)
# check
# sample['retire_month_diff'].notnull() | p(sum) # 181 (192 retired respondents)

# Build training regression
reg_train_1 = smf.ols('retire_month_diff ~ sex_1 + married_1 + grandchild_1 + degree_1 + '
                      'below_degree_1 + a_levels_1 + o_levels_1 + no_qual_1 + early_retire_incentive_1 + pen_db_1 + '
                      'pen_dc_1 + pen_any_1 + gor_1 + adl_1 + poor_health_1 + cardio_1 + noncardio_1',
                      data=sample).fit()
reg_train_1.summary() # R = 0.303, N = 176, Adjusted R = 0.193 (*)

reg_train_2 = smf.ols('retire_month_diff ~ sex_1 + married_1 + grandchild_1 + degree_1 + '
                      'below_degree_1 + a_levels_1 + o_levels_1 + no_qual_1 + early_retire_incentive_1 + pen_db_1 + '
                      'pen_dc_1 + pen_any_1 + gor_1',
                      data=sample).fit()
reg_train_2.summary2() | p(print) # R = 0.294, N = 176, Adjusted R = 0.203 (*)

# reg_train_3 = smf.ols('retire_month_diff ~ sex_1 + married_1 + grandchild_1 + degree_1 + '
#                       'below_degree_1 + a_levels_1 + o_levels_1 + no_qual_1 + early_retire_incentive_1 + pen_db_1 + '
#                       'pen_dc_1 + pen_any_1 + adl_1 + poor_health_1 + cardio_1 '
#                       '+ noncardio_1 + gor_1',
#                       data=sample).fit()
# reg_train_3.summary2() | p(print) # R = 0.320, N = 88, Adjusted R = 0.06

# reg_train_4 = smf.ols('retire_month_diff ~ sex_1 + married_1 + grandchild_1 + degree_1 + '
#                       'below_degree_1 + a_levels_1 + o_levels_1 + no_qual_1 + early_retire_incentive_1 + pen_db_1 + '
#                       'pen_dc_1 + pen_any_1 + total_pension_2002 + total_wealth_1 + adl_1 + poor_health_1 + cardio_1 '
#                       '+ noncardio_1 + gor_1',
#                       data=sample).fit()
# reg_train_4.summary() | p(print) # R = 0.954 (this is abnormally high), N = 28;

# Take a look at the residual distribution
tt = pd.DataFrame({'resid': reg_train_2.resid})
(pn.ggplot(tt, pn.aes('resid')) + pn.geom_histogram(fill='white', colour='black'))

# I will try model 2 first, as it has good adjusted R
np.random.seed(1)
sample['pred_retire_month_diff'] = reg_train_2.predict(sample) + (reg_train_2.resid | p(np.random.choice, size=1)) | p(round)

sample['pred_retire_month'] = sample.apply(
    lambda row: (pd.DateOffset(months=row['pred_retire_month_diff']) + row['birth_month'])
    if not (pd.isna(row['pred_retire_month_diff']) or pd.isna(row['birth_month']))
    else None, 
    axis=1
)

sample['final_retire_month'] = np.where(sample['treatment'] == 0, sample['pred_retire_month'], sample['retire_month'])
# check
pd.isna(sample['final_retire_month']).sum() # there are 36 NAs is the final retirement month, as there are missing values in some of the predictors

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
sample['angina_heart_attack_diabetes_post'] = (sample[['angina_post', 'heart_attack_post', 'diabetes_post']] == 1).any(axis=1).astype(int)
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
sample.to_csv(os.path.join(derived_path, 'sample_simulate_retire.csv'), index=False)

########## Inspection
