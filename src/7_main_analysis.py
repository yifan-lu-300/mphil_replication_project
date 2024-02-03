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
sample = pd.read_csv(os.path.join(derived_path, 'sample_health_stock.csv'))

# Any newly diagnosed disease (post) #TODO: transfer this part to 4_simulate_retirement?
disease_list = ['angina', 'heart_attack', 'stroke', 'diabetes', 'arthritis', 'cancer', 'psych']
sample['any_post'] = (sample[[disease + '_post' for disease in disease_list]] == 1).any(axis=1).astype(int)
# Any newly diagnosed disease (pre)
sample['any_pre'] = (sample[[disease + '_pre' for disease in disease_list]] == 1).any(axis=1).astype(int)
# Newly diagnosed angina, heart attack or stroke (post)
sample['angina_heart_attack_stroke_post'] = (sample[['angina_post', 'heart_attack_post', 'stroke_post']] == 1).any(axis=1).astype(int)
# Any newly diagnosed disease (between w1 and retirement)
sample['any_between'] = ((sample['any_pre'] == 1) & (
    (sample[[disease + '_1' for disease in disease_list]] != 1).all(axis=1))).astype(int)

# ######### PSM lack Lives in deprived region according to index of multiple deprivation (at wave 1) Lives in very
# deprived region according to index of multiple deprivation (at wave 1) Has ever had symptomatic heart attack/angina
# according to Rose questionnaire (at wave 1) ex_work_missing_1, ex_limit_missing_1, bmi_missing_1 and
# ex_alive_missing_1 deleted due to no variation once the corresponding non-missing variables are controlled for

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

# Estimate propensity score with probit regression
xset2_formula = 'treatment ~ ' + ' + '.join(xset2)
xset2_model = smf.probit(xset2_formula, data=sample).fit()

sample['ps_xset2'] = xset2_model.predict(sample)

# IPTW for ATT #TODO: not consistent with original results
# Write a for loop to produce the result table
main_table = pd.DataFrame(np.nan, index=range(18), columns=['Xset1', 'Xset2', 'Xset3'])
y_list = ['any_post', 'angina_heart_attack_stroke_post'] + [disease + '_post' for disease in disease_list]
x_list = [xset1, xset2, xset3]

for i, xset in enumerate(x_list):
    # Estimate propensity score with probit regression
    ps_formula = 'treatment ~ ' + ' + '.join(xset)
    ps_model = smf.probit(formula=ps_formula, data=sample).fit()
    sample[f'ps_xset{i + 1}'] = ps_model.predict(sample)

    # IPTW for ATT
    sample[f'iptw_xset{i + 1}'] = sample.apply(
        lambda x: 1 if x['treatment'] == 1 else x[f'ps_xset{i + 1}'] / (1 - x[f'ps_xset{i + 1}']),
        axis=1)

    # # IPTW for ATE
    # sample[f'iptw_xset{i + 1}'] = sample.apply(
    #     lambda x: 1 / x[f'ps_xset{i + 1}'] if x['treatment'] == 1 else 1 / (1 - x[f'ps_xset{i + 1}']),
    #     axis=1)

    # Estimate ATT
    for j, y in enumerate(y_list):
        att_model = smf.wls(f'{y} ~ treatment', data=sample, weights=sample[f'iptw_xset{i + 1}']).fit()

        # Save results to main_table
        main_table.iloc[j * 2, i] = att_model.params['treatment']
        main_table.iloc[j * 2 + 1, i] = att_model.tvalues['treatment']

for i, xset in enumerate(x_list):
    # Estimate propensity score with probit regression
    ps_formula = 'treatment ~ ' + ' + '.join(xset)
    ps_model = smf.probit(formula=ps_formula, data=sample).fit()
    sample[f'ps_xset{i + 1}'] = ps_model.predict(sample)

    # KNN matching
    treatment_df = sample.loc[sample['treatment'] == 1, ['idauniq', f'ps_xset{i + 1}']].dropna()
    control_df = sample.loc[sample['treatment'] == 0, ['idauniq', f'ps_xset{i + 1}']].dropna()
    control_df['index'] = range(len(control_df))

    k_nn = NearestNeighbors(n_neighbors=4)
    k_nn.fit(X=control_df[[f'ps_xset{i + 1}']])

    # Finding control observations
    matched_k = k_nn.kneighbors(X=treatment_df[[f'ps_xset{i + 1}']], return_distance=False).flatten()

    # Calculate the number of times each matched observation is used
    control_index, control_count = np.unique(matched_k, return_counts=True)
    control_id = pd.DataFrame({'index': control_index, 'count': control_count})

    control_df = pd.merge(control_df, control_id, how='left', on='index')

    matched_id = pd.concat([control_df.loc[control_df['count'].notna(), 'idauniq'],
                            treatment_df['idauniq']]).to_list()

    # Get the final matched sample
    matched_sample = sample.loc[sample['idauniq'].isin(matched_id), :]

    # Create matching weight
    matched_sample = pd.merge(matched_sample, control_df[['idauniq', 'count']], how='left', on='idauniq')
    matched_sample['weight'] = np.where(matched_sample['treatment'] == 1, 1, matched_sample['count'])

    for j, y in enumerate(y_list):
        # model = smf.wls(f'{y} ~ treatment + ps_xset{i + 1}',
        #                 data=matched_sample,
        #                 weights=matched_sample['weight']).fit()

        model = smf.ols(f'{y} ~ treatment + ps_xset{i + 1}',
                        data=matched_sample).fit()

        # Save results to main_table
        main_table.iloc[j * 2, i] = model.params['treatment']
        main_table.iloc[j * 2 + 1, i] = model.tvalues['treatment']


########## Radius matching
treatment_df = sample.loc[sample['treatment'] == 1, ['idauniq', 'ps_xset2']].dropna()
control_df = sample.loc[sample['treatment'] == 0, ['idauniq', 'ps_xset2']].dropna()
control_df['index'] = range(len(control_df))

#TODO: I currently did not use sex_1 and age_2002 as balancing scores
#TODO: I also need to go with k nearest neighbour
# radius = 0.1 * sample['ps_xset2'].std()
# r_nn = NearestNeighbors(radius=radius)
# r_nn.fit(X=control_df[['ps_xset2']])
# matched_r = r_nn.radius_neighbors(X=treatment_df[['ps_xset2']], return_distance=False)

k_nn = NearestNeighbors(n_neighbors=4)
k_nn.fit(X=control_df[['ps_xset2']])
matched_k = k_nn.kneighbors(X=treatment_df[['ps_xset2']], return_distance=False).flatten()

control_index, control_count = np.unique(matched_k, return_counts=True)
control_id = pd.DataFrame({'index': control_index, 'count': control_count})

control_df = pd.merge(control_df, control_id, how='left', on='index')

matched_id = pd.concat([control_df.loc[control_df['count'].notna(), 'idauniq'],
                        treatment_df['idauniq']]).to_list()

matched_sample = sample.loc[sample['idauniq'].isin(matched_id), :]

matched_sample = pd.merge(matched_sample, control_df[['idauniq', 'count']], how='left', on='idauniq')
matched_sample['weight'] = np.where(matched_sample['treatment'] == 1, 1, matched_sample['count'])

nmixx = smf.ols('any_post ~ treatment + ps_xset2', data=matched_sample).fit()
nmixx.summary()






# Any chronic disease
# estimate propensity score using probit regression

# IV
