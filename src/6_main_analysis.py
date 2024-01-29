import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from causalinference import CausalModel
from linearmodels.iv import IV2SLS
import plotnine as pn
from sspipe import p, px

# Set up paths
derived_path = os.path.abspath('..') | p(os.path.join, 'data', 'derived')
# Read data
sample = pd.read_csv(os.path.join(derived_path, 'sample_simulate_retire.csv'))

# Any newly diagnosed disease (post)
disease_list = ['angina', 'heart_attack', 'stroke', 'diabetes', 'arthritis', 'cancer', 'psych']
sample['any_post'] = (sample[[disease + '_post' for disease in disease_list]] == 1).any(axis=1).astype(int)
# Any newly diagnosed disease (pre)
sample['any_pre'] = (sample[[disease + '_pre' for disease in disease_list]] == 1).any(axis=1).astype(int)
# Any newly diagnosed disease (between w1 and retirement)
sample['any_between'] = ((sample['any_pre'] == 1) & ((sample[[disease + '_1' for disease in disease_list]] != 1).all(axis=1))).astype(int)

########## PSM
# lack 
# Lives in deprived region according to index of multiple deprivation (at wave 1) 
# Lives in very deprived region according to index of multiple deprivation (at wave 1)
# Drinks over limit per week (at wave 1)
# Has ever had symptomatic heart attack/angina according to Rose questionnaire (at wave 1)

full_set = ['sex_1', 'age_2002', 'age_2002_squared', 'child_in_house_1', 'london_1', 'outside_uk_1', 'degree_1',
            'job_years_1', 'job_permanent_1', 'job_30h_1', 'reach_spa_2004', 'early_retire_incentive_1', 'pen_db_1',
            'pen_dc_1', 'total_pension_2002_10k', 'ex_work_1', 'ex_work_missing_1', 'poor_health_1', 'cesd_1',
            'limit_1', 'diabetes_hypertension_1', 'angina_heart_attack_stroke_1', 'arthritis_osteoporosis_1',
            'cancer_1', 'psych_1', 'any_between', 'bad_health_0', 'bad_ghq_0', 'condcnt_0', 'high_bp_0', 'smoke_now_1',
            'smoke_past_1', 'no_activities_1', 'ex_limit_1', 'ex_limit_missing_1', 'parents_died_1', 'adl_1',
            'recall_words_1', 'bmi_0', 'bmi_missing_0', 'vitamin_health_0', 'private_health_1', 'ex_alive_1',
            'ex_alive_missing_1']

# Xset 1
xset1 = ['sex_1', 'age_2002', 'age_2002_squared', 'child_in_house_1', 'london_1', 'outside_uk_1', 'degree_1',
         'job_years_1', 'job_permanent_1', 'job_30h_1', 'reach_spa_2004', 'early_retire_incentive_1', 'pen_db_1',
         'pen_dc_1', 'total_pension_2002_10k', 'ex_work_1', 'ex_work_missing_1', 'poor_health_1', 'cesd_1',
         'limit_1', 'diabetes_hypertension_1', 'angina_heart_attack_stroke_1', 'arthritis_osteoporosis_1',
         'cancer_1', 'any_between', 'bad_health_0', 'bad_ghq_0', 'condcnt_0', 'high_bp_0', 'smoke_now_1',
         'smoke_past_1', 'no_activities_1', 'ex_limit_1', 'ex_limit_missing_1']

# Estimate propensity score through probit regression
xset1_ps_f = 'treatment ~ ' + ' + '.join(xset1)
xset1_ps = smf.probit(xset1_ps_f, data=sample).fit()
xset1_ps.summary2() | p(print)

tt = smf.probit('treatment ~ sex_1 + age_2002', data=sample).fit()
# I think it is probably because there are too many variables so the dataset is essentially nan
tt = sample[xset1]
sample['outside_uk_1'].value_counts(dropna=False)
sample['fqcbthr_1'].value_counts(dropna=False)

# Any chronic disease
# estimate propensity score using probit regression

# TODO: CausalModel can only calculate propensity score via logistic regression
psm_x1 = CausalModel(Y=sample['any_post'],
                     D=sample['treatment'],
                     X=sample['sex_1'])

psm_x1.est_via_matching()

sample[xset1].dtypes

# IV
