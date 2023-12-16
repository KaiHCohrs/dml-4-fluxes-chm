import pandas as pd
import os
import sys
import numpy as np
# normalize variables
# compute smooth cycle
# generate synthetic data

def load_data(site_name):
    if site_name == 'book':
        data = pd.read_csv(os.path.join(os.path.dirname(__file__),'../../data/Synthetic4BookChap.csv'))
    elif site_name == 'puechabon':
        data = pd.read_csv(os.path.join(os.path.dirname(__file__),'../../data/Puechabon2008-2009.csv'))
    else:
        folder_name = 'FLX_' + site_name
        
        files = os.listdir(os.path.join(os.path.dirname(__file__),'../../data/' + folder_name))
        for file in files:
            if file.startswith('FLX_' + site_name + '_FLUXNET2015_FULLSET_HH'):
                file_name = file
        data = pd.read_csv(os.path.join(os.path.dirname(__file__),'../../data/' + folder_name + '/' + file_name))
    return data

def unwrap_time(data):
    if 'DateTime' not in data.columns:
        data['DateTime'] = pd.to_datetime(data['TIMESTAMP_START']+15, format="%Y%m%d%H%M")
    data["Date"] = pd.to_datetime(data['DateTime']).dt.date
    data["Time"] = pd.to_datetime(data['DateTime']).dt.time
    data["Month"] = pd.to_datetime(data['Date']).dt.month
    data["Year"] = pd.to_datetime(data['Date']).dt.year
    data["doy"] = pd.to_datetime(data['DateTime']).dt.dayofyear
    return data

def standardize_column_names(data):
    
    '''
        TA_F -> TA
        TA_F_QC -> TA_QC
        SW_IN_F -> SW_IN
        SW_IN_F_QC -> SW_IN_QC
        LW_IN_F -> LW_IN
        LW_IN_F_QC -> LW_IN_QC
        VPD_F -> VPD
        VPD_F_QC -> VPD_QC
        PA_F -> PA
        P_F -> P
        WS_F -> WS
        LE_F_MDS -> LE
        LE_F_MDS_QC -> LE_QC
        TS_F_MDS_1 -> TS_1
        TS_F_MDS_2 -> TS_2
        SWC_F_MDS_1 -> SWC_1
        SWC_F_MDS_1_QC -> SWC_1_QC
        NEE_CUT_USTAR50 -> NEE
        NEE_CUT_USTAR50_QC -> NEE_QC
        NEE_CUT_USTAR50_RANDUNC -> NEE_RANDUNC
        RECO_NT_CUT_USTAR50 -> RECO_NT
        GPP_NT_CUT_USTAR50 -> GPP_NT
        RECO_DT_CUT_USTAR50 -> RECO_DT
        GPP_DT_CUT_USTAR50 -> GPP_DT
    '''
    mappings = {'TA_F': 'TA',
                'TA_F_QC': 'TA_QC',
                'SW_IN_F': 'SW_IN',
                'SW_IN_F_QC': 'SW_IN_QC',
                'LW_IN_F': 'LW_IN',
                'LW_IN_F_QC': 'LW_IN_QC',
                'VPD_F': 'VPD',
                'VPD_F_QC': 'VPD_QC',
                'PA_F': 'PA',
                'PA_F_QC': 'PA_QC',
                'P_F': 'P',
                'P_F_QC': 'P_QC',
                'WS_F': 'WS',
                'LE_F_MDS': 'LE',
                'LE_F_MDS_QC': 'LE_QC',
                'TS_F_MDS_1': 'TS_1',
                'TS_F_MDS_1_QC': 'TS_1_QC',
                'TS_F_MDS_2': 'TS_2',
                'TS_F_MDS_2_QC': 'TS_2_QC',
                'SWC_F_MDS_1': 'SWC_1',
                'SWC_F_MDS_1_QC': 'SWC_1_QC',
                'SWC_F_MDS_2': 'SWC_2',
                'SWC_F_MDS_2_QC': 'SWC_2_QC',
                'NEE_CUT_USTAR50': 'NEE',
                'NEE_CUT_USTAR50_QC': 'NEE_QC',
                'NEE_CUT_USTAR50_RANDUNC': 'NEE_RANDUNC',
                'RECO_NT_CUT_USTAR50': 'RECO_NT',
                'GPP_NT_CUT_USTAR50': 'GPP_NT',
                'RECO_DT_CUT_USTAR50': 'RECO_DT',
                'GPP_DT_CUT_USTAR50': 'GPP_DT'}
    
    for old, new in mappings.items():
        #if old in data.columns and new not in data.columns:
        if old in data.columns:
            data[new] = data[old]
    return data    
    
def w_def_cum(data):
    P = data['P']
    LE = data['LE']    
    if 'P' in data.columns and 'LE' in data.columns:
        n = len(LE)
        ET = LE / 2.45e6 * 1800
        wdefCum = np.zeros(n)
        wdefCum[1:] = np.NaN
        
        for i in range(1,n):
            wdefCum[i] = np.minimum(wdefCum[i-1] + P[i-1] - ET[i-1],0)
            
        if np.isnan(wdefCum[i]):
            wdefCum[i] = wdefCum[i-1]
        data['wdefcum'] = wdefCum
        
    else:
        print('You are missing either P or LE to compute wdefcum')   
    return data

def diffuse_to_direct_rad(data):
    data['SW_ratio'] = 1 - data['SW_IN']/(data['SW_IN_POT']+1e-10)
    return data

def quality_check(data, variables):
    '''
    Introduce yearwise quality_flag on the data
    The criterion are:
    1. percentage of meteorological gap-filled data is less than 20%
    2. measured NEE covered at least 10% of both daytime and nighttime periods
    '''
    data['QC'] = 0
    
    
    for year in data['Year'].unique():
        data_year = data[data['Year'] == year]
        fail_count = 0
        for var in variables:
            if (var+'_QC') in data.columns and var != 'NEE':
                if sum(data_year[var + '_QC'] == 0)/len(data_year) <= 0.8:
                    fail_count += 1
            elif var != 'NEE':
                print(var + 's quality_flag does not exist')
                
        night_data = data_year[data_year['NIGHT'] == 1]
        day_data = data_year[data_year['NIGHT'] == 0]
        
        if sum(night_data['NEE_QC'] == 0)/len(night_data) < 0.1:
            fail_count += 1
        if sum(day_data['NEE_QC'] == 0)/len(day_data) < 0.1:
            fail_count += 1
        data.loc[data['Year'] == year,'QC'] = fail_count
        
    return data

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def sw_pot_sm(data):
    data['SW_POT_sm'] = moving_average(data['SW_IN_POT'], 480)
    return data

def sw_pot_sm_diff(data):
    SW_POT_sm = data['SW_POT_sm']
    SW_POT_sm_diff = np.hstack((np.array(SW_POT_sm[1]-SW_POT_sm[0]), (np.roll(SW_POT_sm,-1) - SW_POT_sm)[1:]))
    data['SW_POT_sm_diff'] = moving_average(10000*SW_POT_sm_diff, 480)
    return data

def synthetic_dataset(data, Q10, relnoise):
    
    if 'SW_POT_sm' not in data.columns:
        data = sw_pot_sm(data)
        
    if 'SW_POT_sm_diff' not in data.columns:
        data = sw_pot_sm_diff(data)
        
    SW_IN = data['SW_IN']
    SW_POT_sm = data['SW_POT_sm']
    SW_POT_sm_diff = data['SW_POT_sm_diff']
    TA = data['TA']
    VPD = data['VPD']
    
    RUE_syn = 0.5 * np.exp(-(0.1*(TA-20))**2) * np.minimum(1, np.exp(-0.1 * (VPD-10)))
    GPP_syn = RUE_syn * SW_IN / 12.011
    Rb_syn = SW_POT_sm * 0.01 - SW_POT_sm_diff * 0.005
    Rb_syn = 0.75 * (Rb_syn - np.nanmin(Rb_syn) + 0.1*np.pi)
    RECO_syn = Rb_syn * Q10 ** (0.1*(TA-15.0))
    NEE_syn_clean = RECO_syn - GPP_syn 
    NEE_syn = NEE_syn_clean * (1 + relnoise * np.random.normal(size=NEE_syn_clean.size))
    
    data['RUE_syn'] = RUE_syn
    data['GPP_syn'] = GPP_syn
    data['Rb_syn'] = Rb_syn
    data['RECO_syn'] = RECO_syn
    data['NEE_syn_clean'] = NEE_syn_clean
    data['NEE_syn'] = NEE_syn
    
    return data

def normalize(data, variables, yearly=True, norm_type='standardize'):
    
    if yearly:
        for var in variables:
            data[var + '_n'] = 0

        for Year in data['Year'].unique():
            indices = (data['Year']==Year)
            for var in variables:
                if norm_type == 'standardize':
                    data.loc[indices, var + '_n'] = (data.loc[indices, var] - data.loc[indices, var].mean()) / data.loc[indices, var].std()
                elif norm_type == 'in_one':
                    max = data.loc[indices, var].abs().max()
                    min = -max
                    data.loc[indices, var + '_n'] = 2 * ((data.loc[indices, var] - min) / (max-min) -0.5)
    else:
        for var in variables:
            data[var + '_n'] = 0
            
        for var in variables:
            if norm_type == 'standardize':
                data[var + '_n'] = (data[var] - data[var].mean()) / data[var].std()
            elif norm_type == 'in_one':
                max = data[var].abs().max()
                min = -max
                data[var + '_n'] = 2 * ((data[var] - min) / (max-min) -0.5)
                
    return data