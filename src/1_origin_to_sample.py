import os
import pandas as pd
import numpy as np
from sspipe import p

# set up paths
origin_path = os.path.abspath('..') | p(os.path.join, 'data', 'tab')
derived_path = os.path.abspath('..') | p(os.path.join, 'data', 'derived')

# select variables
vars_0 = ['idauniq', 'genhelf2', 'ghqg2', 'condcnt', 'bmival', 'vitamin'] + \
    [f'illsm{number}' for number in range(1, 7)]

vars_1_core = ['idauniq', 'indsex', 'indager', 'indobyr', 'couple1', 'digran', 'fqcbthr', 'apobr', 'edqual',
               'wpsjoby', 'wpsjobm', 'wpcjob', 'wphjob', 'wphwrk', 'wperet'] + \
    ['eligw1', 'askpx1', 'iintdtm', 'iintdty', 'intdaty', 'gor', 'chinhh1', 'heacta', 'heactb', 'cflisen', 'wpphi'] + \
    [f'heada0{number}' for number in range(1,10)] + ['heada10', 'heada11'] + \
    ['hehelf', 'hehelfb', 'hegenh', 'hegenhb', 'heill', 'helim'] + \
    [f'hedim0{number}' for number in range(1,8)] + [f'hedib0{number}' for number in range(1,10)] + ['hedib10'] + \
    ['psceda', 'pscedb', 'pscedc', 'pscedd', 'pscede', 'pscedf', 'pscedg', 'pscedh'] + \
    ['exlo80', 'expw', 'exhlim'] + \
    [f'wpact{number}' for number in range(1,7)] + ['wpdes', 'wplljy', 'wplljm']

vars_1_pw = ['idauniq', 'pen_db', 'pen_dc', 'pen_any', 'pripenw1_2002', 'statepenw1_2002']
vars_1_f = ['idauniq', 'nettotw_bu_s', 'empinc_r_s']
vars_1_ifs = ['idauniq', 'smoker', 'smokerstat', 'malive', 'falive', 'spage', 'llsill'] + \
    ['hemobwa', 'hemobsi', 'hemobch', 'hemobcs', 'hemobcl', 'hemobst', 'hemobre', 'hemobpu', 'hemobli', 'hemobpi', 'hemob96']

vars_2_core = ['idauniq', 'Hehelf', 'Heill', 'Helim'] + \
    [f"heada0{number}" for number in range(1,10)] + ['heada10'] + \
    [f'wpact{number}' for number in range(1,7)] + ['wpdes', 'wplljy', 'wplljm', 'wpactw'] + \
    [f"hedim0{number}" for number in range(1,9)] + \
    ['HeagaR', 'HeagaRY', # angina
     'HeAgbR', 'HeAgbRY', # heart attack
     'HeAgeR', 'HeAgeRY', # stroke
     'HeAgdR', 'HeAgdRY', # diabetes
     'HeAgfR', 'HeAgfRY', # arthritis
     'HeAggR', 'HeAggRY', # cancer
     'HeAghR', 'HeAghRY'] # psychiatric

vars_3_core = ['idauniq', 'hegenh', 'heill', 'helim'] + \
    ['hemobwa', 'hemobsi', 'hemobch', 'hemobcs', 'hemobcl', 'hemobst', 'hemobre', 'hemobpu', 'hemobli', 'hemobpi'] + \
    ['wpactpw', 'wpactse', 'wpdes', 'wplljy', 'wplljm', 'wpactw'] + \
    ['dhediman', 'heagar', 'heagary', 'hediman', # angina
     'dhedimmi', 'heagbr', 'heagbry', 'hedimmi', # heart attack
     'dhedimst', 'heager', 'heagery', 'hedimst', # stroke
     'dhedimdi', 'heagdr', 'heagdry', 'hedimdi', # diabetes
     'dhedibar', 'heagfr', 'heagfry', 'hedibar', # arthritis
     'dhedibca', 'heaggr', 'heaggry', 'hedibca', # cancer
     'dhedibps', 'heaghr', 'heaghry', 'hedibps'] # psychiatric

# sample selection (following the procedure in working paper Table A1)
wave_1_core = pd.read_table(os.path.join(origin_path, 'wave_1_core_data_v3.tab'),
                            low_memory=False,
                            usecols=vars_1_core)
wave_1_core = wave_1_core.loc[wave_1_core['eligw1'] == 1, :] # sample member (all age >= 50), 11391 (11392)
wave_1_core = wave_1_core.loc[wave_1_core['askpx1'] != 1, :] # not interviewed by proxy, -158 (-158)

activities_1 = [f'wpact{number}' for number in range(1,7)]
activities_1_pw = (wave_1_core[activities_1] == 1).any(axis=1)
wave_1_core = wave_1_core.loc[activities_1_pw & (wave_1_core['wpdes'] == 2), :] # employed at wave 1, 2906 (2906)

wave_1_pw = pd.read_table(os.path.join(origin_path, 'wave_1_pension_wealth_v2.tab'),
                          low_memory=False,
                          usecols=vars_1_pw)
wave_1_f = pd.read_table(os.path.join(origin_path, 'wave_1_financial_derived_variables.tab'),
                         low_memory=False,
                         usecols=vars_1_f)
wave_1_ifs = pd.read_table(os.path.join(origin_path, 'wave_1_ifs_derived_variables.tab'),
                           low_memory=False,
                           usecols=vars_1_ifs)
wave_1_full = (pd.merge(wave_1_core,
                       wave_1_pw,
                       on='idauniq',
                       how='inner') | p(pd.merge,
                                        wave_1_f,
                                        on='idauniq',
                                        how='inner')) | p(pd.merge, 
                                                          wave_1_ifs,
                                                          on='idauniq',
                                                          how='inner')

wave_2_core = pd.read_table(os.path.join(origin_path, 'wave_2_core_data_v4.tab'),
                            low_memory=False,
                            usecols=vars_2_core)

wave_12 = pd.merge(wave_1_full.rename(columns=lambda x: x if x == 'idauniq' else x + '_1'),
                   wave_2_core.rename(columns=lambda x: x if x == 'idauniq' else x + '_2'),
                   how='inner',
                   on='idauniq') # wave 2 attrition, 2369 (2369)

activities_2 = [f'wpact{number}_2' for number in range(1,7)]
activities_2_pw = (wave_12[activities_2] == 1).any(axis=1)

employed_2 = activities_2_pw & (wave_12['wpdes_2'] == 2)
sum(employed_2) # employed at wave 2, 1803 (1803)
retired_2 = (wave_12['wpactw_2'] != 1) & (wave_12['wpdes_2'] == 1)
sum(retired_2) # retired at wave 2, 268 (268)

wave_12 = wave_12.loc[employed_2 | retired_2, :] # 2071 (2071)

wave_3_core = pd.read_table(os.path.join(origin_path, 'wave_3_elsa_data_v4.tab'),
                            low_memory=False,
                            usecols=vars_3_core)

wave_123 = pd.merge(wave_12,
                    wave_3_core.rename(columns=lambda x: x if x == 'idauniq' else x + '_3'),
                    how='inner',
                    on='idauniq') # wave 3 attrition, 1769 (1769)

activities_23_pw = ((wave_123[activities_2] == 1).any(axis=1)) & (wave_123['wpactpw_3'] == 1)
activities_23_none = (wave_123['wpactw_2'] != 1) & (wave_123['wpactw_3'] != 1)

employed_23 = activities_23_pw & (wave_123['wpdes_2'] == 2) & (wave_123['wpdes_3'] == 2)
sum(employed_23) # employed at wave 2 and 3, 1247 (1247)
retired_23 = activities_23_none & (wave_123['wpdes_2'] == 1) & (wave_123['wpdes_3'] == 1)
sum(retired_23) # retired at wave 2 and 3, 192 (192)

sample_123 = wave_123.loc[employed_23 | retired_23, :] # final sample, 1439 (1439)

wave_0 = pd.read_table(os.path.join(origin_path, 'wave_0_common_variables_v2.tab'),
                       low_memory=False,
                       usecols=vars_0)

sample = pd.merge(sample_123,
                  wave_0.rename(columns=lambda x: x if x == 'idauniq' else x + '_0'),
                  how='left',
                  on='idauniq')

employed_id = wave_123.loc[employed_23, 'idauniq']
sample['treatment'] = np.where(sample['idauniq'].isin(employed_id), 0, 1) # D = 1 for retired individuals
sample['treatment'].value_counts()

########## Save data
sample.to_csv(os.path.join(derived_path, 'sample_uncleaned.csv'), index=False)

########## Inspection