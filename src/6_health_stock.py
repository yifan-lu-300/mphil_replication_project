import os
import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
import plotnine as pn
from sspipe import p, px

# Set up paths
derived_path = os.path.abspath('..') | p(os.path.join, 'data', 'derived')
# Read data
sample = pd.read_csv(os.path.join(derived_path, 'sample_simulate_retire.csv'))

# ########## Inspection
# sample['self_health_1'].value_counts(dropna=False)
# sample['genhelf_0'].value_counts(dropna=False)

########## Wave 1
hs_data_1 = sample[
    ['idauniq', 'self_health_1', 'sex_1', 'age_2002', 'degree_1', 'below_degree_1', 'a_levels_1', 'o_levels_1',
     'no_qual_1', 'job_permanent_1', 'job_years_1', 'job_30h_1', 'hemobwa_1', 'hemobsi_1', 'hemobch_1', 'hemobcs_1',
     'hemobcl_1', 'hemobst_1', 'hemobre_1', 'hemobpu_1', 'hemobli_1', 'hemobpi_1', 'memory_1', 'eye_1', 'hear_1',
     'walk_1', 'angina_1', 'heart_attack_1', 'stroke_1', 'diabetes_1', 'arthritis_1', 'cancer_1', 'psych_1',
     'hypertension_1', 'asthma_1', 'cesd_1', 'smoke_now_1', 'drink_now_1', 'no_activities_1']].dropna()

hs_probit_1 = OrderedModel(hs_data_1['self_health_1'],
                           hs_data_1[['sex_1', 'age_2002', 'degree_1', 'below_degree_1', 'a_levels_1', 'o_levels_1',
                                      'no_qual_1', 'job_permanent_1', 'job_years_1', 'job_30h_1',
                                      'hemobwa_1', 'hemobsi_1', 'hemobch_1', 'hemobcs_1', 'hemobcl_1', 'hemobst_1',
                                      'hemobre_1', 'hemobpu_1', 'hemobli_1', 'hemobpi_1',
                                      'memory_1', 'eye_1', 'hear_1', 'walk_1',
                                      'angina_1', 'heart_attack_1', 'stroke_1', 'diabetes_1', 'arthritis_1', 'cancer_1',
                                      'psych_1', 'hypertension_1', 'asthma_1',
                                      'cesd_1',
                                      'smoke_now_1', 'drink_now_1', 'no_activities_1']],
                           distr='probit').fit(method='bfgs')  # TODO: try different methods

probs_1 = hs_probit_1.predict() | p(pd.DataFrame)
hs_data_1['health_stock_1'] = probs_1.apply(lambda row: row[0] * 1 + row[1] * 2 + row[2] * 3 + row[3] * 4 + row[4] * 5,
                                            axis=1)
(pn.ggplot(hs_data_1, pn.aes(x='self_health_1', y='health_stock_1')) +
 pn.geom_point() +
 pn.geom_smooth(method='lm', se=False) +
 pn.theme_bw())

# merge back into sample
sample = pd.merge(sample, hs_data_1[['idauniq', 'health_stock_1']], how='left', on='idauniq')

########## Wave 0
hs_data_0 = sample[
    ['idauniq', 'genhelf_0', 'sex_1', 'age_0', 'degree_0', 'below_degree_0', 'a_levels_0', 'o_levels_0', 'no_qual_0',
     'job_ft_0', 'job_employ_0', 'eye_0', 'hear_0', 'high_bp_0', 'heart_attack_angina_0', 'stroke_0',
     'diabetes_0', 'arthritis_0', 'cancer_0', 'psych_depress_0', 'asthma_0', 'smoke_now_0', 'dnnow_0']].dropna()

hs_probit_0 = OrderedModel(hs_data_0['genhelf_0'],
                           hs_data_0[['sex_1', 'age_0', 'degree_0', 'below_degree_0', 'a_levels_0', 'o_levels_0', 'no_qual_0',
                                      'job_ft_0', 'job_employ_0', 'eye_0', 'hear_0', 'high_bp_0', 'heart_attack_angina_0', 'stroke_0',
                                      'diabetes_0', 'arthritis_0', 'cancer_0', 'psych_depress_0', 'asthma_0', 'smoke_now_0', 'dnnow_0']],
                           distr='probit').fit(method='bfgs')

probs_0 = hs_probit_0.predict() | p(pd.DataFrame)
hs_data_0['health_stock_0'] = probs_0.apply(lambda row: row[0] * 1 + row[1] * 2 + row[2] * 3 + row[3] * 4 + row[4] * 5,
                                            axis=1)
(pn.ggplot(hs_data_0, pn.aes(x='genhelf_0', y='health_stock_0')) +
 pn.geom_point() +
 pn.geom_smooth(method='lm', se=False) +
 pn.theme_bw())

# merge back into sample
sample = pd.merge(sample, hs_data_0[['idauniq', 'health_stock_0']], how='left', on='idauniq')

########## Save data
sample.to_csv(os.path.join(derived_path, 'sample_health_stock.csv'), index=False)
