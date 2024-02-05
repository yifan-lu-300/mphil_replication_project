import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from causalinference import CausalModel
from linearmodels.iv import IV2SLS
import plotnine as pn
from sspipe import p, px

# Set up paths
derived_path = os.path.abspath('..') | p(os.path.join, 'data', 'derived')
# Read data
sample = pd.read_csv(os.path.join(derived_path, 'sample_health_stock_ext.csv'))

# ######### PSM lack Lives in deprived region according to index of multiple deprivation (at wave 1) Lives in very
# deprived region according to index of multiple deprivation (at wave 1) Has ever had symptomatic heart attack/angina
# according to Rose questionnaire (at wave 1) ex_work_missing_1, ex_limit_missing_1, bmi_missing_1 and
# ex_alive_missing_1 deleted due to no variation once the corresponding non-missing variables are controlled for

disease_list = ['angina', 'heart_attack', 'diabetes', 'stroke', 'arthritis', 'cancer', 'psych']

full_set = ['sex_1', 'age_2002', 'age_2002_squared', 'child_in_house_1', 'london_1', 'outside_uk_1', 'degree_1',
            'job_years_1', 'job_permanent_1', 'job_30h_1', 'reach_spa_2004', 'early_retire_incentive_1', 'pen_db_1',
            'pen_dc_1', 'total_pension_2002_10k', 'ex_work_1', 'ex_work_missing_1', 'poor_health_1', 'cesd_1',
            'limit_1', 'diabetes_hypertension_1', 'angina_heart_attack_stroke_1', 'arthritis_osteoporosis_1',
            'cancer_1', 'psych_1', 'any_between', 'bad_health_0', 'bad_ghq_0', 'condcnt_0', 'high_bp_0', 'smoke_now_1',
            'smoke_past_1', 'drink_over_1', 'no_activities_1', 'ex_limit_1', 'ex_limit_missing_1', 'parents_died_1',
            'adl_1',
            'recall_words_1', 'bmi_0', 'bmi_missing_0', 'vitamin_health_0', 'private_health_1', 'ex_alive_1',
            'ex_alive_missing_1']

# Xsets
xset1 = ['sex_1', 'age_2002', 'age_2002_squared', 'child_in_house_1', 'outside_uk_1', 'degree_1',
         'job_years_1', 'job_permanent_1', 'job_30h_1', 'london_1', 'reach_spa_2004', 'early_retire_incentive_1',
         'pen_db_1', 'pen_dc_1', 'total_pension_2002_10k', 'ex_work_1',
         'poor_health_1', 'cesd_1', 'limit_1', 'diabetes_hypertension_1', 'angina_heart_attack_stroke_1',
         'arthritis_osteoporosis_1', 'cancer_1', 'any_between', 'bad_health_0', 'bad_ghq_0', 'condcnt_0', 'high_bp_0',
         'smoke_now_1', 'smoke_past_1', 'drink_over_1', 'no_activities_1', 'ex_limit_1']

xset2 = ['sex_1', 'age_2002', 'age_2002_squared', 'child_in_house_1', 'outside_uk_1', 'degree_1',
         'job_years_1', 'job_permanent_1', 'job_30h_1', 'london_1', 'reach_spa_2004', 'early_retire_incentive_1',
         'pen_db_1', 'pen_dc_1', 'total_pension_2002_10k', 'ex_work_1',
         'poor_health_1', 'cesd_1', 'limit_1', 'diabetes_hypertension_1', 'angina_heart_attack_stroke_1',
         'arthritis_osteoporosis_1', 'cancer_1', 'psych_1', 'any_between', 'bad_health_0', 'bad_ghq_0', 'condcnt_0',
         'high_bp_0', 'smoke_now_1', 'smoke_past_1', 'drink_over_1', 'no_activities_1', 'ex_limit_1', 'parents_died_1',
         'adl_1',
         'recall_words_1', 'bmi_0', 'private_health_1', 'ex_alive_1']

xset3 = ['sex_1', 'age_2002', 'age_2002_squared', 'child_in_house_1', 'outside_uk_1', 'degree_1',
         'job_years_1', 'job_permanent_1', 'job_30h_1', 'london_1', 'reach_spa_2004', 'early_retire_incentive_1',
         'pen_db_1', 'pen_dc_1', 'total_pension_2002_10k', 'ex_work_1', 'any_between', 'health_stock_1',
         'health_stock_0']

# # check whether xset2 has abnormally high NA rate
# tt = sample[xset2].apply(lambda x: x.isna().sum() / len(sample), axis=0)
# # I intentionally removed vitamin_health_0 due to its high NA rate 18%
#
# # check whether xset3 has abnormally high NA rate
# tt = sample[xset3].apply(lambda x: x.isna().sum() / len(sample), axis=0)

# Calculate propensity score
ps_formula = f'treatment ~ ' + ' + '.join(xset1)
ps_model = smf.probit(formula=ps_formula, data=sample).fit()
sample['ps_xset1'] = ps_model.predict(sample)

y_list = ['any_post', 'angina_heart_attack_diabetes_post'] + [disease + '_post' for disease in disease_list]
x_list = [xset1, xset2, xset3]

########## Simple OLS
ps_table = pd.DataFrame(np.nan, index=range(18), columns=['Xset1', 'Xset2', 'Xset3'])
# Write a for loop to produce the result table
for i, xset in enumerate(x_list):
    for j, y in enumerate(y_list):
        model_formula = f'{y} ~ treatment + ' + ' + '.join(xset)
        model = smf.ols(formula=model_formula, data=sample).fit()

        # Save results to main_table
        ps_table.iloc[j * 2, i] = model.params['treatment']
        ps_table.iloc[j * 2 + 1, i] = model.tvalues['treatment']

########## IV
# Select neighbourhood around state pension age (women 60 men 65)
sample_iv = sample.loc[((sample['sex_1'] == 1) & (sample['age_2002'].isin([56, 57, 58, 59, 60]))) |
                       ((sample['sex_1'] == 0) & (sample['age_2002'].isin([61, 62, 63, 64, 65]))), :]

iv_table = pd.DataFrame(np.nan, index=range(18), columns=['Xset4'])
# For loop
for j, y in enumerate(y_list):
    model_data = sample_iv[[f'{y}', 'sex_1', 'ex_work_1', 'ex_limit_1', 'treatment', 'spage_2']].dropna(subset=[f'{y}'])
    model_formula = f'{y} ~ 1 + C(sex_1) + C(ex_work_1) + C(ex_limit_1) + [treatment ~ spage_2]'
    model = IV2SLS.from_formula(model_formula, model_data).fit()

    iv_table.iloc[j * 2, 0] = model.params['treatment']
    iv_table.iloc[j * 2 + 1, 0] = model.params['treatment'] / model.std_errors['treatment']

########## Concatenate the two tables together
main_table = pd.concat([ps_table, iv_table], axis=1)

########## Save the table
main_table.to_csv(os.path.join(derived_path, 'main_table_ext.csv'), index=False)
