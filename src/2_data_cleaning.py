import os
import pandas as pd
import numpy as np
from sspipe import p

# set up paths
origin_path = os.path.abspath('..') | p(os.path.join, 'data', 'tab')
derived_path = os.path.abspath('..') | p(os.path.join, 'data', 'derived')
# read data
sample = pd.read_csv(os.path.join(derived_path, 'sample_uncleaned.csv'), low_memory=False)

########## Data cleaning - Descriptive stats
# sex
# sample['indsex_1'].value_counts(dropna=False) # no NAs
sample['sex_1'] = np.select(condlist=[sample['indsex_1'] == 2, sample['indsex_1'] == 1],
                            choicelist=[1, 0],
                            default=np.nan)

# age in year 2002 #TODO both, maybe use year of household interview (currently, year of individual interview)
# sample['iintdty_1'].value_counts(dropna=False) # no NAs
sample['age_2002'] = np.select(condlist=[sample['iintdty_1'] == 2002, sample['iintdty_1'] == 2003],
                               choicelist=[sample['indager_1'], sample['indager_1'] - 1],
                               default=np.nan)

# married or cohabiting
# sample['couple1_1'].value_counts(dropna=False) # no NAs
sample['married_1'] = np.select(condlist=[sample['couple1_1'].isin([1, 2]), sample['couple1_1'] == 3],
                                choicelist=[1, 0],
                                default=np.nan)

# has grandchildren #TODO both
# sample['digran_1'].value_counts(dropna=False) # NAs present
sample['grandchild_1'] = np.select(condlist=[sample['digran_1'] == 1, sample['digran_1'] == 2],
                                   choicelist=[1, 0],
                                   default=np.nan)

# birth outside uk #TODO retired, probably because I took information from two variables
# pd.crosstab(sample['fqcbthr_1'], sample['apobr_1']) # they complement each other
sample['outside_uk_1'] = np.select(condlist=[sample['fqcbthr_1'] == 2, sample['fqcbthr_1'] == 1, sample['apobr_1'] == 2, sample['apobr_1'] == 1],
                                   choicelist=[1, 0, 1, 0],
                                   default=np.nan)

# university degree
# sample['edqual_1'].value_counts(dropna=False) # no NAs
sample['degree_1'] = np.where(sample['edqual_1'] == 1, 1, 0)
# Higher education below degree
sample['below_degree_1'] = np.where(sample['edqual_1'] == 2, 1, 0)
# A levels
sample['a_levels_1'] = np.where(sample['edqual_1'] == 3, 1, 0)
# O levels
sample['o_levels_1'] = np.where(sample['edqual_1'] == 4, 1, 0)
# No qualification
sample['no_qual_1'] = np.where(sample['edqual_1'] == 7, 1, 0)

# Number of years in current job #TODO employed, maybe can also use wpsjobm_1 (month)
# (sample['wpsjoby_1'] < 0).sum() # NAs present
sample['job_years_1'] = np.where(sample['wpsjoby_1'] > 0, sample['iintdty_1'] - sample['wpsjoby_1'], np.nan)

# Current job is permanent #TODO retired
# sample['wpcjob_1'].value_counts(dropna=False) # NAs present
sample['job_permanent_1'] = np.select(condlist=[sample['wpcjob_1'] == 4, sample['wpcjob_1'].isin([1, 2, 3])],
                                      choicelist=[1, 0],
                                      default=np.nan)

# 1-30h work per week at current job #TODO retired
# (sample['wphjob_1'] < 0).sum() # NAs present
sample['job_30h_1'] = np.select(condlist=[sample['wphjob_1'].between(1, 30), sample['wphjob_1'] > 30],
                                choicelist=[1, 0],
                                default=np.nan)

# will reach state pension age at wave 2 (men 65 women 60) #TODO both
sample['reach_spa_2004'] = np.select(condlist=[(sample['sex_1'] == 0) & (sample['age_2002'] + 2 >= 65),
                                               (sample['sex_1'] == 0) & (sample['age_2002'] + 2 < 65),
                                               (sample['sex_1'] == 1) & (sample['age_2002'] + 2 >= 60),
                                               (sample['sex_1'] == 1) & (sample['age_2002'] + 2 < 65)],
                                     choicelist=[1, 0, 1, 0],
                                     default=np.nan)

# early retirement incentives
# sample['wperet_1'].value_counts(dropna=False) # NAs present
sample['early_retire_incentive_1'] = np.select(condlist=[sample['wperet_1'] == 1, sample['wperet_1'] == 2],
                                               choicelist=[1, 0],
                                               default=np.nan)
# defined benefit pension
# sample['pen_db_1'].value_counts() # no NAs

# defined contribution pension
# sample['pen_dc_1'].value_counts() # no NAs

# any private pension scheme #TODO both: possibly due to not removing NAs
# sample['pen_any_1'].value_counts() # NAs present
sample['pen_private_1'] = np.where(sample['pen_any_1'] == -8, np.nan, sample['pen_any_1'])

# total pension wealth in 2002
# (sample['pripenw1_2002_1'] < 0).sum() # no NAs
# (sample['statepenw1_2002_1'] < 0).sum() # no NAs
sample['total_pension_2002'] = sample['pripenw1_2002_1'] + sample['statepenw1_2002_1']

# net (non-pension) total wealth #TODO both
# (sample['nettotw_bu_s_1'].isin([' '])).sum()
# (sample['nettotw_bu_s_1'].isin(['-999.0', '-998.0', '-995.0'])).sum()
sample['total_wealth_1'] = sample['nettotw_bu_s_1'].apply(lambda x: np.nan if x == ' ' else pd.to_numeric(x))

# employment income per week #TODO 
# (sample['empinc_r_s_1'] < 0).sum() # no NAs

# number of difficulties in ADL
adl_1 = [f'heada0{number}_1' for number in range(1, 10)] + ['heada10_1', 'heada11_1']
# sample[adl_1].apply(lambda x: x.isin([-9, -8, -1])).all(axis=1).sum() # no NAs
sample['adl_1'] = ((sample[adl_1] >= 1) & (sample[adl_1] <= 10)).sum(axis=1)

# self-assessed health poor or fair
pd.crosstab(sample['hegenh_1'], sample['hehelf_1']) # so they complement each other
sample['poor_health_1'] = np.select(condlist=[sample['hehelf_1'].isin([4, 5]), sample['hehelf_1'].isin([1, 2, 3]), sample['hegenh_1'].isin([3, 4, 5]), sample['hegenh_1'].isin([1, 2])],
                                    choicelist=[1, 0, 1, 0],
                                    default=np.nan)

# has diagnosed cardio disease #TODO retired
cardio_1 = [f'hedim0{number}_1' for number in range(1, 8)]
# sample[cardio_1].apply(lambda x: x.isin([-9, -8, -1])).all(axis=1).sum() # no NAs
sample['cardio_1'] = ((sample[cardio_1] >= 1) & (sample[cardio_1] <= 8)).any(axis=1)

# has diagnosed non-cardio disease #TODO both: possibly because the hedib list contains too many additional diseases?
noncardio_1 = [f'hedib0{number}_1' for number in range(1,10)] + ['hedib10_1']
# sample[noncardio_1].apply(lambda x: x.isin([-9, -8, -1])).all(axis=1).sum() # no NAs
sample['noncardio_1'] = ((sample[noncardio_1] >= 1) & (sample[noncardio_1] <= 9)).any(axis=1)

# likelihood of being alive #TODO both: possibly due to not removing NAs
# sample['exlo80_1'].value_counts() # NAs present
sample['ex_alive_1'] = np.where(sample['exlo80_1'] < 0, np.nan, sample['exlo80_1']/100)

# likelihood of working #TODO both: possibly due to not removing NAs
# sample['expw_1'].value_counts() # NAs present
sample['ex_work_1'] = np.where(sample['expw_1'] < 0, np.nan, sample['expw_1']/100)

# likelihood that health limits ability to work #TODO both: possibly due to not removing NAs
# sample['exhlim_1'].value_counts() # NAs present
sample['ex_limit_1'] = np.where(sample['exhlim_1'] < 0, np.nan, sample['exhlim_1']/100)

########## Data cleaning - health after retirement table
# poor self-assessed health
# sample['hegenh_3'].value_counts() # NAs present
sample['poor_health_3'] = np.select(condlist=[sample['hegenh_3'].isin([4, 5]), sample['hegenh_3'].isin([1, 2, 3])],
                                    choicelist=[1, 0],
                                    default=np.nan)

# limiting long-standing illness
# sample['helim_3'].value_counts() # NAs present
sample['limit_3'] = np.select(condlist=[sample['helim_3'] == 1, sample['helim_3'] == 2],
                              choicelist=[1, 0],
                              default=np.nan)

# number of ADLs
adl_3 = ['hemobwa_3', 'hemobsi_3', 'hemobch_3', 'hemobcs_3', 'hemobcl_3', 'hemobst_3', 'hemobre_3', 'hemobpu_3', 'hemobli_3', 'hemobpi_3']
# sample[adl_3].apply(lambda x: x < 0).all(axis=1).sum() # no NAs
sample['adl_3'] = (sample[adl_3] == 1).sum(axis=1)

########## Data cleaning - Retirement month
sample['last_job_year_2'] = np.where(sample['wplljy_2'] < 0, None, sample['wplljy_2'].astype(str))
sample['last_job_month_2'] = np.where(sample['wplljm_2'] < 0, None, sample['wplljm_2'].astype(str))
sample['last_job_2'] = sample['last_job_year_2'] + '-' + sample['last_job_month_2']

sample['last_job_year_3'] = np.where(sample['wplljy_3'] < 0, None, sample['wplljy_3'].astype(str))
sample['last_job_month_3'] = np.where(sample['wplljm_3'] < 0, None, sample['wplljm_3'].astype(str))
sample['last_job_3'] = sample['last_job_year_3'] + '-' + sample['last_job_month_3']

# sample.loc[sample['last_job_3'].notnull(), ['last_job_3', 'last_job_2', 'treatment', 'wpdes_2', 'wpdes_3']]
# We can see there are respondents who retired twice. I believe this is because some respondents picked up another job
# after wave 2 and quit the job again before wave 3, so it may be better to use the second retirement date as the
# retirement date
sample['retire_month'] = np.where(sample['last_job_3'].notnull(), sample['last_job_3'], sample['last_job_2']) | p(pd.to_datetime)
sample['retire_month'][20] | p(type)

########## Data cleaning - Disease month
# no -8 and -9, only -1 (not applicable)
def clean_disease(name, wave, year_var, month_var):
    sample[f'{name}_year_{wave}'] = np.where(sample[year_var] < 0, None, sample[year_var].astype(str))
    sample[f'{name}_month_{wave}'] = np.where(sample[month_var] < 0, None, sample[month_var].astype(str))
    sample[f'{name}_{wave}'] = (sample[f'{name}_year_{wave}'] + '-' + sample[f'{name}_month_{wave}']) | p(pd.to_datetime)

clean_disease(name='angina', wave=2, year_var='HeagaRY_2', month_var='HeagaR_2')
clean_disease(name='heart_attack', wave=2, year_var='HeAgbRY_2', month_var='HeAgbR_2')
clean_disease(name='stroke', wave=2, year_var='HeAgeRY_2', month_var='HeAgeR_2')
clean_disease(name='diabetes', wave=2, year_var='HeAgdRY_2', month_var='HeAgdR_2')
clean_disease(name='arthritis', wave=2, year_var='HeAgfRY_2', month_var='HeAgfR_2')
clean_disease(name='cancer', wave=2, year_var='HeAggRY_2', month_var='HeAggR_2')
clean_disease(name='psych', wave=2, year_var='HeAghRY_2', month_var='HeAghR_2')

clean_disease(name='angina', wave=3, year_var='heagary_3', month_var='heagar_3')
clean_disease(name='heart_attack', wave=3, year_var='heagbry_3', month_var='heagbr_3')
clean_disease(name='stroke', wave=3, year_var='heagery_3', month_var='heager_3')
clean_disease(name='diabetes', wave=3, year_var='heagdry_3', month_var='heagdr_3')
clean_disease(name='arthritis', wave=3, year_var='heagfry_3', month_var='heagfr_3')
clean_disease(name='cancer', wave=3, year_var='heaggry_3', month_var='heaggr_3')
clean_disease(name='psych', wave=3, year_var='heaghry_3', month_var='heaghr_3')

########## Data cleaning - Wave 1 diseases
# no NAs in cardio_1 and noncardio_1
sample['angina_1'] = (sample[cardio_1] == 2).any(axis=1).astype(int)
sample['heart_attack_1'] = (sample[cardio_1] == 3).any(axis=1).astype(int)
sample['stroke_1'] = (sample[cardio_1] == 8).any(axis=1).astype(int)
sample['diabetes_1'] = (sample[cardio_1] == 7).any(axis=1).astype(int)
sample['arthritis_1'] = (sample[noncardio_1] == 3).any(axis=1).astype(int)
sample['cancer_1'] = (sample[noncardio_1] == 5).any(axis=1).astype(int)
sample['psych_1'] = (sample[cardio_1] == 7).any(axis=1).astype(int)

########## Data cleaning - main analysis
# squared age (in year 2002)
sample['age_2002_squared'] = sample['age_2002'] ** 2

# child lives in household (at wave 1)
# sample['chinhh1_1'].value_counts() # no NAs
sample['child_in_house_1'] = np.select(condlist=[sample['chinhh1_1'] == 1, sample['chinhh1_1'] == 2],
                                       choicelist=[1, 0],
                                       default=np.nan)

# lives in London (at wave 1)
# sample['gor_1'].value_counts(dropna=False) # no NAs
sample['london_1'] = np.where(sample['gor_1'] == 'H', 1, 0)

# lives in deprived region according to index of multiple deprivation (at wave 1) #TODO
# lives in very deprived region according to index of multiple deprivation (at wave 1) #TODO

# total pension wealth/10000 (at wave 1)
sample['total_pension_2002_10k'] = sample['total_pension_2002'] / 10000

# missing likelihood of working (at wave 1)
# sample['expw_1'].value_counts(dropna=False) # NAs present
sample['ex_work_missing_1'] = np.where(sample['expw_1'] < 0, 1, 0)

# has depression according to CES depression scale (at wave 1)
cesd_list_1 = ['psceda_1', 'pscedb_1', 'pscedc_1', 'pscedd_1', 'pscede_1', 'pscedf_1', 'pscedg_1', 'pscedh_1']
# (sample[cesd_list_1] < 0).any(axis=1).sum() # NAs present
# reverse the score of positive items
sample['pscedd_1'] = sample['pscedd_1'].replace({1: 2, 2: 1})
sample['pscedf_1'] = sample['pscedf_1'].replace({1: 2, 2: 1})
sample['cesd_1'] = np.select(condlist=[(sample[cesd_list_1] < 0).any(axis=1), (sample[cesd_list_1] == 1).sum(axis=1) >= 3],
                             choicelist=[np.nan, 1],
                             default=0)

# has limiting long-standing illness (at wave 1)
# sample['heill_1'].value_counts() # NAs present
# sample['helim_1'].value_counts() # NAs present
sample['limit_1'] = np.select(condlist=[(sample['heill_1'] == 1) & (sample['helim_1'] == 1), sample['heill_1'] == 2, (sample['heill_1'] == 1) & (sample['helim_1'] == 2)],
                              choicelist=[1, 0, 0],
                              default=np.nan)

# no all NAs in cardio & noncardio
# diabetes and/or hypertension (at wave 1)
sample['diabetes_hypertension_1'] = (sample[cardio_1].isin([1, 7])).any(axis=1).astype(int)
# angina, heart attack or stroke
sample['angina_heart_attack_stroke_1'] = (sample[cardio_1].isin([2, 3, 8])).any(axis=1).astype(int)
# arthritis or osteoporosis
sample['arthritis_osteoporosis_1'] = (sample[noncardio_1].isin([3, 4])).any(axis=1).astype(int)

# bad or very bad self-assessed health (at HSE interview)
# sample['genhelf2_0'].value_counts(dropna=False) # NAs present
sample['bad_health_0'] = np.select(condlist=[sample['genhelf2_0'] == 3, sample['genhelf2_0'].isin([1, 2])],
                                   choicelist=[1, 0],
                                   default=np.nan)

# bad general health according to GHQ12 (at HSE interview)
# sample['ghqg2_0'].value_counts(dropna=False) # NAs present
sample['bad_ghq_0'] = np.select(condlist=[sample['ghqg2_0'] == 3, sample['ghqg2_0'].isin([1, 2])],
                                choicelist=[1, 0],
                                default=np.nan)

# number of long-standing illnesses that have been diagnosed (at HSE interview)
# sample['condcnt_0'].value_counts(dropna=False) # NAs present & directly usable

# hypertensive blood pressure (at HSE interview)
illness_0 = [f'illsm{number}_0' for number in range(1, 7)]
(sample[illness_0] <= -2).all(axis=1).sum() # no all NAs
sample['high_bp_0'] = np.where((sample[illness_0] == 17).any(axis=1), 1, 0)

# current smoker (at wave 1)
# sample['smoker_1'].value_counts(dropna=False) # no NAs
sample['smoke_now_1'] = np.select(condlist=[sample['smoker_1'] == 1, sample['smoker_1'] == 0],
                                  choicelist=[1, 0],
                                  default=np.nan)

# ex-smoker (at wave 1)
# sample['smokerstat_1'].value_counts(dropna=False) # no NAs
sample['smoke_past_1'] = np.select(condlist=[sample['smokerstat_1'].isin([1, 2, 3]), sample['smokerstat_1'].isin([0, 4])],
                                   choicelist=[1, 0],
                                   default=np.nan)

# drinks over limit per week (at wave 1) #TODO

# never engages activities neither vigorous nor moderate (at wave 1)
# sample['heacta_1'].value_counts(dropna=False) # no NAs
sample['no_activities_1'] = np.select(condlist=[((sample['heacta_1'] == 4) & (sample['heactb_1'] == 4)), ((sample['heacta_1'] < 0) | (sample['heactb_1'] < 0))],
                                      choicelist=[1, np.nan],
                                      default=0)

# missing likelihood that health limits ability to work (at wave 1)
sample['exhlim_1'].value_counts(dropna=False) # NAs present
sample['ex_limit_missing_1'] = np.where(sample['exhlim_1'] < 0, 1, 0)

# both parents dead (at wave 1)
sample['dinma_1'].value_counts(dropna=False) # NAs present
sample['parents_died_1'] = np.select(condlist=[((sample['dinma_1'] == 2) & sample['dinfa_1'] == 2), ((sample['dinma_1'] < 0) | (sample['dinfa_1'] < 0))],
                                     choicelist=[1, np.nan],
                                     default=0)

# number of words that could be recalled according to cognitive function test (at wave 1)
sample['cflisen_1'].value_counts(dropna=False) # NAs present
sample['recall_words_1'] = np.where(sample['cflisen_1'] < 0, np.nan, sample['cflisen_1'])

# has ever had symptomatic heart attack/angina according to Rose questionnaire (at wave 1) #TODO

# body mass index over 35 (at HSE nurse interview)
# sample['bmival_0'].value_counts(dropna=False) # NAs present
sample['bmi_0'] = np.select(condlist=[sample['bmival_0'] >= 35, (sample['bmival_0'] < 0) | (sample['bmival_0'].isna())],
                            choicelist=[1, np.nan],
                            default=0)
# missing body mass index (at HSE nurse interview)
sample['bmi_missing_0'] = np.where((sample['bmival_0'] < 0) | (sample['bmival_0'].isna()), 1, 0)

# Takes vitamins, minerals to improve health (at HSE interview)
# sample['vitamin_0'].value_counts() # NAs present
sample['vitamin_health_0'] = np.select(condlist=[sample['vitamin_0'] == 1, sample['vitamin_0'] == 2],
                                       choicelist=[1, 0],
                                       default=np.nan)

# has private health insurance (at wave 1)
# sample['wpphi_1'].value_counts() # NAs present
sample['private_health_1'] = np.select(condlist=[sample['wpphi_1'].isin([1, 2]), sample['wpphi_1'] == 3],
                                       choicelist=[1, 0],
                                       default=np.nan)

# missing likelihood of being alive in next 10 years (at wave 1)
sample['exlo80_1'].value_counts(dropna=False) # NAs present
sample['ex_alive_missing_1'] = np.where(sample['exlo80_1'] < 0, 1, 0)

########## Save data
sample.to_csv(os.path.join(derived_path, 'sample_cleaned.csv'), index=False)

########## Inspection
