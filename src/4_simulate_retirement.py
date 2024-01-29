import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import plotnine as pn
from sspipe import p, px

# Set up paths
origin_path = os.path.abspath('..') | p(os.path.join, 'data', 'tab')
derived_path = os.path.abspath('..') | p(os.path.join, 'data', 'derived')

# Read data
sample = pd.read_csv(os.path.join(derived_path, 'sample_cleaned.csv'), low_memory=False, parse_dates=['retire_month']) # datetime info will not be preserved in to_csv

# Inspection
sample['treatment'].value_counts()
sample['outside_uk_1'].value_counts(dropna=False)

sample.loc[sample['angina_2'].notna() | sample['angina_3'].notna(), ['treatment', 'angina_2', 'angina_3']]
sample.loc[sample['stroke_2'].notna() | sample['stroke_3'].notna(), ['treatment', 'stroke_2', 'stroke_3']] # there are cases dated back to 1995

# Approach 1: Regression
# Assume people are all born in July
sample['birth_month'] = (sample['indobyr_1'].astype(str) + '-6') | p(pd.to_datetime)

# calculate difference in month
sample['retire_month_diff'] = (sample['retire_month'].dt.to_period('M') - sample['birth_month'].dt.to_period('M')).apply(lambda x: x.n if pd.notnull(x) else None)
# check
sample['retire_month_diff'].notnull() | p(sum) # 181 (192 retired respondents)

# Build training regression
reg_train_1 = smf.ols('retire_month_diff ~ sex_1 + married_1 + grandchild_1 + degree_1 + outside_uk_1 + '
                      'below_degree_1 + a_levels_1 + o_levels_1 + no_qual_1 + early_retire_incentive_1 + pen_db_1 + '
                      'pen_dc_1 + pen_any_1 + gor_1',
                      data=sample).fit()
reg_train_1.summary2() | p(print) # R = 0.293, N = 175, Adjusted R = 0.196 (*)

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
# reg_train_4.summary() | p(print) # R = 0.954 (this is abnormally high), N = 28; After removing outside_uk R = 0.322 and N = 88

# Take a look at the residual distribution
tt = pd.DataFrame({'resid': reg_train_2.resid})
(pn.ggplot(tt, pn.aes('resid')) + pn.geom_histogram(fill='white', colour='black'))

# I will try model 2 first, as it has good adjusted R
np.random.seed(1)
sample['pred_month_diff_2'] = reg_train_2.predict(sample) + (reg_train_2.resid | p(np.random.choice, size=1)) | p(round)

sample['pred_month_2'] = sample.apply(
    lambda row: (pd.DateOffset(months=row['pred_month_diff_2']) + row['birth_month'])
    if not (pd.isna(row['pred_month_diff_2']) or pd.isna(row['birth_month'])) 
    else None, 
    axis=1
)

sample['final_month_2'] = np.where(sample['treatment'] == 0, sample['pred_month_2'], sample['retire_month'])
# check
pd.isna(sample['final_month_2']).sum() # there are 39 NAs is the final retirement month, as there are missing values in some of the predictors

# Approach 2: PSM 

# Approach 3: More advanced machine learning methods (with training/test sets)
# The main problem is that I lack birth month data

########## Disease date vs. Retirement date
# no NAs in angina_1, and NAs in angina_2 and angina_3 all mean 'not applicable' rather than e.g. refusal
def disease_pre_retire(row, name):
    if pd.isna(row[f'{name}_2']) & pd.isna(row[f'{name}_3']):
        return np.where(row[f'{name}_1'] == 1, 1, 0)
    elif pd.isna(row['final_month_2']):
        return None
    elif pd.notna(row[f'{name}_2']):
        return np.where(pd.to_datetime(row[f'{name}_2']) <= row['final_month_2'], 1, 0)
    elif pd.notna(row[f'{name}_3']):
        return np.where(pd.to_datetime(row[f'{name}_3']) <= row['final_month_2'], 1, 0)

sample['angina_pre'] = sample.apply(disease_pre_retire, name='angina', axis=1)
sample['heart_attack_pre'] = sample.apply(disease_pre_retire, name='heart_attack', axis=1)
sample['stroke_pre'] = sample.apply(disease_pre_retire, name='stroke', axis=1)
sample['diabetes_pre'] = sample.apply(disease_pre_retire, name='diabetes', axis=1)
sample['arthritis_pre'] = sample.apply(disease_pre_retire, name='arthritis', axis=1)
sample['cancer_pre'] = sample.apply(disease_pre_retire, name='cancer', axis=1)
sample['psych_pre'] = sample.apply(disease_pre_retire, name='psych', axis=1)

# check
sample.loc[sample['angina_pre'] == 1, ['treatment', 'angina_1', 'angina_2', 'angina_3', 'final_month_2']]
sample.loc[sample['psych_pre'] == 1, ['treatment', 'psych_1', 'psych_2', 'psych_3', 'final_month_2']]

def disease_post_retire(row, name):
    if row[f'{name}_1'] == 1:
        return 0
    elif pd.isna(row[f'{name}_2']) & pd.isna(row[f'{name}_3']):
        return 0
    elif pd.isna(row['final_month_2']):
        return None
    elif pd.notna(row[f'{name}_2']):
        return np.where(pd.to_datetime(row[f'{name}_2']) > row['final_month_2'], 1, 0)
    elif pd.notna(row[f'{name}_3']):
        return np.where(pd.to_datetime(row[f'{name}_3']) > row['final_month_2'], 1, 0)

sample['angina_post'] = sample.apply(disease_post_retire, name='angina', axis=1)
sample['heart_attack_post'] = sample.apply(disease_post_retire, name='heart_attack', axis=1)
sample['stroke_post'] = sample.apply(disease_post_retire, name='stroke', axis=1)
sample['diabetes_post'] = sample.apply(disease_post_retire, name='diabetes', axis=1)
sample['arthritis_post'] = sample.apply(disease_post_retire, name='arthritis', axis=1)
sample['cancer_post'] = sample.apply(disease_post_retire, name='cancer', axis=1)
sample['psych_post'] = sample.apply(disease_post_retire, name='psych', axis=1)

# check
sample.loc[sample['angina_post'] == 1, ['treatment', 'angina_1', 'angina_2', 'angina_3', 'final_month_2']]
sample.loc[sample['psych_post'] == 1, ['treatment', 'psych_1', 'psych_2', 'psych_3', 'final_month_2']] # observations may be too fewer

########## Save data
sample.to_csv(os.path.join(derived_path, 'sample_simulate_retire.csv'), index=False)