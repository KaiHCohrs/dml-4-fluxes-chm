import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import dml4fluxes.datasets.relevant_variables as relevant_variables
from pathlib import Path
from dml4fluxes.datasets.preprocessing import load_data, unwrap_time, standardize_column_names,\
                                                daily_means, NEE_quality_masks
from dml4fluxes.datasets.generate_data import synthetic_dataset
                                                
from dml4fluxes.experiments.utility import get_available_sites
import json

def timely_averages(data):
    for flux_name in ['RECO_DT', 'RECO_NT', 'RECO_orth', 'RECO_ann',
                        'GPP_DT', 'GPP_NT', 'GPP_orth', 'GPP_ann',
                        'NEE', 'NEE_orth', 'NEE_ann']:
        try:
            yearly_avg = data.groupby(['site', 'Year'])[flux_name].mean()
            yearly_avg = yearly_avg.rename(flux_name + '_yearly_avg')
            data = data.join(yearly_avg, on=['site', 'Year'])

            daily_avg = data.groupby(['site', 'Year', 'doy'])[flux_name].mean()
            daily_avg = daily_avg.rename(flux_name + '_daily_avg')
            data = data.join(daily_avg, on=['site', 'Year', 'doy'])
        except:
            pass

        data['soy'] = data['doy'].apply(lambda x: x//5)
        seasonal_avg = data.groupby(['site', 'Year', 'soy'])[flux_name].mean()
        seasonal_avg = seasonal_avg.rename(flux_name + '_seasonal_avg')
        data = data.join(seasonal_avg, on=['site', 'Year', 'soy'])
        
        data[flux_name + '_daily_anomalies_avg'] = data[flux_name + '_daily_avg'] - data[flux_name + '_seasonal_avg']
        data[flux_name + '_seasonal_without_yearly_avg'] = data[flux_name + '_seasonal_avg'] - data[flux_name + '_yearly_avg']

    return data

def numerical_cross_consistency(experiment_name, syn=False, year=2015, biome=False, sites=False, new=False, relnoise=0.4, partial=False):

    results_path = Path(__file__).parent.parent.parent.joinpath("results",
                                                        experiment_name)
    
    if partial:
        if f'results_NEE_{partial}.csv' in os.listdir(results_path) and not new:
            tables = dict()
            for file in results_path.glob('*'):
                if file.name.startswith('results'):
                    tables[file.name[8:-4]] = pd.read_csv(file).drop("Unnamed: 0", axis=1)
            return tables
    else:
        if 'results_NEE.csv' in os.listdir(results_path) and not new:
            tables = dict()
            for file in results_path.glob('*'):
                if file.name.startswith('results'):
                    tables[file.name[8:-4]] = pd.read_csv(file).drop("Unnamed: 0", axis=1)
            return tables

    
    with open(results_path.joinpath("dataset_config.txt"), 'r') as f:
        dataset_config = json.load(f)
        relnoise = dataset_config['relnoise']

        
    if biome:
        sites = relevant_variables.biomes[biome]
    elif sites:
        pass
    else:
        path = Path(__file__).parent.parent.parent.joinpath("data", 
                                                            f"Fluxnet-{year}",
                                                            "DMLPartitioning",
                                                            experiment_name)
        sites = get_available_sites(path)
        
    all_results = dict()
    all_results['GPP'] = dict()
    all_results['RECO'] = dict()
    all_results['NEE'] = dict()
    all_results['GPP_T'] = dict()
    
    if syn:
        for flux in all_results.values():
            flux['R2'] = dict()
            flux['RMSE'] = dict()
            flux['bias'] = dict()

            for measure in flux.values():
                measure['syn_orth'] = list()
                measure['syn_clean_orth'] = list()

        for site in sites:
            data = load_partition(experiment_name, site, year, syn=True, relnoise=relnoise)
            data['DateTime'] = data.index
            data = unwrap_time(data)
            data = standardize_column_names(data)
            data = data[list(set(data.columns)
                                    & set(relevant_variables.variables))]
            
            data = data.replace(-9999, np.nan)
            
            if partial == 'DT':
                data = data.loc[(data['NIGHT'] == 0),:]
            elif partial == 'NT':
                data = data.loc[(data['NIGHT'] == 1),:]
            
            # Drop irrelavant columns
            for flux_name, flux_dict in all_results.items():
                if flux_name == 'GPP_T':
                    flux_name = 'GPP'
                    data['GPP_orth'] = -data['T']
                    
                years = data['Year'].unique()
                for site_year in years:
                    subset = data.loc[(data["Year"] == site_year),:]
                    if sum(~np.isnan(subset[flux_name + '_orth'])) == 0:
                        continue
                    
                    indices = (~np.isnan(subset[flux_name + '_orth']))\
                        & (~np.isnan(subset[flux_name + '_syn']))\
                    # Potentially additional mask
                    if sum(indices) == 0:
                        continue
                    
                    # Compute R2
                    methods = [['syn', 'orth']]
                    if flux_name == 'NEE':
                        methods += [['syn_clean', 'orth']]
                    
                    for method1, method2 in methods:
                        flux_dict['R2'][f'{method1}_{method2}'].append(
                                r2_score(subset[indices][f'{flux_name}_{method1}'],
                                        subset[indices][f'{flux_name}_{method2}']))

                        flux_dict['RMSE'][f'{method1}_{method2}'].append(
                                mean_squared_error(subset[indices][f'{flux_name}_{method1}'],
                                        subset[indices][f'{flux_name}_{method2}'], squared=False))

                        flux_dict['bias'][f'{method1}_{method2}'].append(
                                np.mean(subset[indices][f'{flux_name}_{method1}'])
                                - np.mean(subset[indices][f'{flux_name}_{method2}']))

        tables = dict()
        for flux_name, flux_dict in all_results.items():
            methods = [['syn', 'orth']]
            if flux_name == 'NEE':
                methods += [['syn_clean', 'orth']]
            tables[flux_name] = pd.DataFrame(columns = ["method 1", "method 2"] + list(flux_dict.keys()))
            for i, [method1, method2] in enumerate(methods):
                row = [method1, method2]
                for measure in ['R2', 'RMSE', 'bias']:        
                    data_list = flux_dict[measure][f'{method1}_{method2}']
                    median = np.quantile(data_list, 0.5).round(2)
                    quantile1 = np.quantile(data_list, 0.25).round(2)
                    quantile3 = np.quantile(data_list, 0.75).round(2)
                    row.append(f'{median} ({quantile1}/{quantile3})')
                tables[flux_name].loc[i] = row
    else:
        
        for flux in all_results.values():
            flux['R2'] = dict()
            flux['RMSE'] = dict()
            flux['bias'] = dict()
            flux['R2_rm_daily'] = dict()
            flux['RMSE_rm_daily'] = dict()

            for measure in flux.values():
                measure['DT_orth'] = list()
                measure['DT_ann'] = list()
                measure['NT_orth'] = list()
                measure['NT_ann'] = list()
                measure['DT_NT'] = list()
                measure['ann_orth'] = list()

        for site in sites:
            print(site)
            data = load_partition(experiment_name, site, year)
            data['DateTime'] = data.index
            data = unwrap_time(data)
            data = standardize_column_names(data)
            #data['GPP_orth'] = -data['T']
            #data['NEE_T'] = data['T'] + data['RECO_orth']
            
            data = data[list(set(data.columns)
                                    & set(relevant_variables.variables))]
            data = NEE_quality_masks(data)
            
            data['NEE_NT'] = -data['GPP_NT'] + data['RECO_NT']
            data['NEE_DT'] = -data['GPP_DT'] + data['RECO_DT']
            data = data.replace(-9999, np.nan)
            if partial == 'DT':
                data = data.loc[(data['NIGHT'] == 0),:]
            elif partial == 'NT':
                data = data.loc[(data['NIGHT'] == 1),:]
            
            data = daily_means(data)

            # Drop irrelavant columns
            for flux_name, flux_dict in all_results.items():
                if flux_name == 'GPP_T':
                    flux_name = 'GPP'
                    data['GPP_orth'] = -data['T']
                    data['GPP_daily_avg'] = -data['T_daily_avg']
                    data['GPP_orth_rm_daily'] = -data['T_rm_daily']
                years = data['Year'].unique()
                for site_year in years:
                    subset = data.loc[(data["Year"] == site_year),:]
                    if sum(~np.isnan(subset[flux_name + '_orth'])) == 0:
                        continue
                    
                    indices = (subset['QM_daily'] == 1)\
                        & (~np.isnan(subset[flux_name + '_orth']))\
                        & (~np.isnan(subset[flux_name + '_NT']))\
                        & (~np.isnan(subset[flux_name + '_DT']))\
                        & (~np.isnan(subset[flux_name + '_ann']))\
                        & (subset['MeasurementNEE_mask']!=0)\
                        & (subset['NEE_QC'] == 0)
                    # Potentially additional mask
                    if sum(indices) == 0:
                        continue
                    
                    # Compute R2
                    for method1, method2 in [['DT', 'orth'], ['DT', 'ann'], ['NT', 'orth'],
                                            ['NT', 'ann'], ['DT', 'NT'], ['ann', 'orth']]:
                        flux_dict['R2'][f'{method1}_{method2}'].append(
                                r2_score(subset[indices][f'{flux_name}_{method1}'],
                                        subset[indices][f'{flux_name}_{method2}']))

                        flux_dict['RMSE'][f'{method1}_{method2}'].append(
                                mean_squared_error(subset[indices][f'{flux_name}_{method1}'],
                                        subset[indices][f'{flux_name}_{method2}'], squared=False))

                        flux_dict['bias'][f'{method1}_{method2}'].append(
                                np.mean(subset[indices][f'{flux_name}_{method1}'])
                                - np.mean(subset[indices][f'{flux_name}_{method2}']))

                        flux_dict['R2_rm_daily'][f'{method1}_{method2}'].append(
                                r2_score(subset[indices][f'{flux_name}_{method1}_rm_daily'],
                                        subset[indices][f'{flux_name}_{method2}_rm_daily']))

                        flux_dict['RMSE_rm_daily'][f'{method1}_{method2}'].append(
                                mean_squared_error(subset[indices][f'{flux_name}_{method1}_rm_daily'],
                                        subset[indices][f'{flux_name}_{method2}_rm_daily'], squared=False))
                        if flux_name == 'NEE' and method1 == 'NT' and method2 == 'ann':
                            print(site_year)
                            print(mean_squared_error(subset[indices][f'{flux_name}_{method1}'],
                                        subset[indices][f'{flux_name}_{method2}'], squared=False))
                            print(r2_score(subset[indices][f'{flux_name}_{method1}'],
                                        subset[indices][f'{flux_name}_{method2}']))
                        
        tables = dict()
        for flux_name, flux_dict in all_results.items():
            tables[flux_name] = pd.DataFrame(columns = ["method 1", "method 2"] + list(flux_dict.keys()))
            for i, [method1, method2] in enumerate([['DT', 'orth'], ['DT', 'ann'], ['NT', 'orth'],
                                            ['NT', 'ann'], ['DT', 'NT'], ['ann', 'orth']]):    
                
                row = [method1, method2]
                for measure in ['R2', 'RMSE', 'bias', 'R2_rm_daily', 'RMSE_rm_daily']:        
                    data_list = flux_dict[measure][f'{method1}_{method2}']
                    median = np.quantile(data_list, 0.5).round(2)
                    quantile1 = np.quantile(data_list, 0.25).round(2)
                    quantile3 = np.quantile(data_list, 0.75).round(2)
                    row.append(f'{median} ({quantile1}/{quantile3})')
                tables[flux_name].loc[i] = row        
    
    if partial:
        for flux, table in tables.items():
            table.to_csv(results_path.joinpath(f'results_{flux}_{partial}.csv'))      
    else:    
        for flux, table in tables.items():
            table.to_csv(results_path.joinpath(f'results_{flux}.csv'))      
        
    return tables, all_results


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def parition_cycles(data):
    for flux_name in ['RECO_DT', 'RECO_NT', 'RECO_orth', 'RECO_orth_res',
                        'GPP_DT', 'GPP_NT', 'GPP_orth',
                        'NEE_DT', 'NEE', 'NEE_orth']:
        
        mean = data.groupby('doy').mean()[flux_name]
        mean_dict = mean.to_dict()
        data[flux_name + '_daily'] = data['doy'].map(mean_dict)
        
        mean = data.groupby('Year').mean()[flux_name]
        mean_dict = mean.to_dict()
        data[flux_name + '_yearly'] = data['Year'].map(mean_dict)
        
        data[flux_name + '_seasonal'] = moving_average(data[flux_name], 240)
        
        data[flux_name + '_anomalies'] = data[flux_name + '_daily'] - data[flux_name + '_seasonal']

        data[flux_name + '_seasonal_mod_yearly'] = moving_average(data[flux_name], 240) - data[flux_name + '_yearly']
        
        for months in [[1,2], [3.4], [5,6], [7,8], [9,10], [11,12]]:
            indices = data['Month'].isin(months)
            subdata = data[data['Month'].isin(months)]
            
            mean = subdata.groupby('Time').mean()[flux_name]
            mean_dict = mean.to_dict()
            data.loc[indices, flux_name + '_diurnal'] = subdata['Time'].map(mean_dict)
        
        return data


def load_partition(experiment_name, site_name, year=2015, old=False, syn=False, relnoise=0):
    data, path = load_data(site_name, year=year, add_ann=True, files_path=True)
    if syn:
        data = synthetic_dataset(data, Q10=1.5, 
                                relnoise=relnoise, 
                                version='simple', 
                                pre_computed=True, 
                                site_name=site_name)
    
    data = unwrap_time(data)
    data = data.set_index('DateTime')
    #data['DateTime'] = data.index
    data = standardize_column_names(data)
    
    folder_name = path.name
    
    if old:
        folder_path = path.parent.joinpath('DMLPartitioning', folder_name,
                                            experiment_name)
    else:    
        folder_path = path.parent.joinpath('DMLPartitioning', experiment_name, 
                                        folder_name)
    partition = pd.read_csv(folder_path.joinpath('orth_partitioning.csv'))
    partition['DateTime'] = pd.to_datetime(partition['DateTime'], format = "%Y-%m-%d %H:%M:%S")
    partition = partition.set_index('DateTime')
    
    for column in partition.columns:
        partition[column] = partition[column].astype(float)
    
    data = pd.merge(data, partition,how='left', left_index=True, right_index=True)

    print('Filter out according to relevant_variables.')
    data = data[list(set(data.columns) & set(relevant_variables.variables))]
    
    return data
    

def evaluate(data, syn=True, yearly=True, part='all', method='orth'):
    if part=='day':
        indices = data['NIGHT']==0
    elif part=='night':
        indices = data['NIGHT']==1
    else:
        indices = [True] * len(data)

    if method != 'orth':
        data[f'RECO_{method}_res'] = data[f'RECO_{method}']

    if syn: 
        results_R2 = pd.DataFrame(columns=['Year', 'RECO', 'RECO_res', 'GPP', 'NEE', 'NEE_clean', 'QC'])
        results_MSE = pd.DataFrame(columns=['Year','RECO', 'RECO_res', 'GPP', 'NEE', 'NEE_clean', 'QC'])

        for year in data['Year'].unique():
            yearly_indices = indices & (data['Year']==year) & (~np.isnan(data[f'RECO_{method}'])) & (~np.isnan(data[f'GPP_{method}']))
            #subdata = partdata            
            
            R2 = dict(Year = year,
                                RECO = r2_score(data.loc[yearly_indices, 'RECO_syn'], data.loc[yearly_indices, f'RECO_{method}']),
                                RECO_res = r2_score(data.loc[yearly_indices, 'RECO_syn'], data.loc[yearly_indices, f'RECO_{method}_res']),
                                GPP = r2_score(data.loc[yearly_indices, 'GPP_syn'], data.loc[yearly_indices, f'GPP_{method}']),
                                NEE = r2_score(data.loc[yearly_indices, 'NEE_syn'], data.loc[yearly_indices, f'NEE_{method}']),
                                NEE_clean = r2_score(data.loc[yearly_indices, 'NEE_syn_clean'], data.loc[yearly_indices, f'NEE_{method}']),
                                QC = data.loc[yearly_indices, 'QC'].unique()[0]
                                )
            MSE = dict(Year = year,
                                RECO = mean_squared_error(data.loc[yearly_indices, 'RECO_syn'], data.loc[yearly_indices, f'RECO_{method}'], squared=False),
                                RECO_res = mean_squared_error(data.loc[yearly_indices, 'RECO_syn'], data.loc[yearly_indices, f'RECO_{method}_res'], squared=False),
                                GPP = mean_squared_error(data.loc[yearly_indices, 'GPP_syn'], data.loc[yearly_indices, f'GPP_{method}'], squared=False),
                                NEE = mean_squared_error(data.loc[yearly_indices, 'NEE_syn'], data.loc[yearly_indices, f'NEE_{method}'], squared=False),
                                NEE_clean = mean_squared_error(data.loc[yearly_indices, 'NEE_syn_clean'], data.loc[yearly_indices, f'NEE_{method}'], squared=False),
                                QC = data.loc[yearly_indices, 'QC'].unique()[0]
                                )
                
            results_R2 = results_R2.append(R2, ignore_index = True)
            results_MSE = results_MSE.append(MSE, ignore_index = True)
            
        condensed = pd.DataFrame(columns=['R2_med', 'R2_25', 'R2_75', 'MSE_med', 'MSE_25', 'MSE_75'])
        condensed['R2_med'] = results_R2.quantile(0.5)
        condensed['R2_25'] = results_R2.quantile(0.25)
        condensed['R2_75'] = results_R2.quantile(0.75)
        condensed['MSE_med'] = results_MSE.quantile(0.5)
        condensed['MSE_25'] = results_MSE.quantile(0.25)
        condensed['MSE_75'] = results_MSE.quantile(0.75)
        #condensed = condensed.drop(['Year', 'QC'])
        
        return results_R2, results_MSE, condensed

    else:
        results_R2 = pd.DataFrame(columns=['Year', 'RECO', 'RECO_res', 'GPP', 'NEE', 'QC', 'method'])
        results_MSE = pd.DataFrame(columns=['Year','RECO', 'RECO_res', 'GPP', 'NEE', 'QC', 'method'])

        for year in data['Year'].unique():
            yearly_indices = indices & (data['Year']==year) & (~np.isnan(data[f'RECO_{method}'])) & (~np.isnan(data[f'GPP_{method}']))
            for methods in ['DT', 'NT', 'DTNT']:
                if methods in ['DT', 'NT']:
                    R2 = dict(Year = year,
                                        RECO = r2_score(data.loc[yearly_indices, 'RECO_'+methods], data.loc[yearly_indices, f'RECO_{method}']),
                                        RECO_res = r2_score(data.loc[yearly_indices, 'RECO_'+methods], data.loc[yearly_indices, f'RECO_{method}_res']),
                                        GPP = r2_score(data.loc[yearly_indices, 'GPP_'+methods], data.loc[yearly_indices, f'GPP_{method}']),
                                        NEE = r2_score(data.loc[yearly_indices, 'NEE_'+methods], data.loc[yearly_indices, f'NEE_{method}']),
                                        QC = data.loc[yearly_indices, 'QC'].unique()[0],
                                        method = methods
                                        )
                    MSE = dict(Year = year,
                                        RECO = mean_squared_error(data.loc[yearly_indices, 'RECO_'+methods], data.loc[yearly_indices, f'RECO_{method}'], squared=False),
                                        RECO_res = mean_squared_error(data.loc[yearly_indices, 'RECO_'+methods], data.loc[yearly_indices, f'RECO_{method}_res'], squared=False),
                                        GPP = mean_squared_error(data.loc[yearly_indices, 'GPP_'+methods], data.loc[yearly_indices, f'GPP_{method}'], squared=False),
                                        NEE = mean_squared_error(data.loc[yearly_indices, 'NEE_'+methods], data.loc[yearly_indices, f'NEE_{method}'], squared=False),
                                        QC = data.loc[yearly_indices, 'QC'].unique()[0],
                                        method = methods
                                        )
                    results_R2 = results_R2.append(R2, ignore_index = True)
                    results_MSE = results_MSE.append(MSE, ignore_index = True)

                else:
                    R2 = dict(Year = year,
                                        RECO = r2_score(data.loc[yearly_indices, 'RECO_DT'], data.loc[yearly_indices, 'RECO_NT']),
                                        RECO_res = r2_score(data.loc[yearly_indices, 'RECO_DT'], data.loc[yearly_indices, 'RECO_NT']),
                                        GPP = r2_score(data.loc[yearly_indices, 'GPP_DT'], data.loc[yearly_indices, 'GPP_NT']),
                                        NEE = r2_score(data.loc[yearly_indices, 'NEE_DT'], data.loc[yearly_indices, 'NEE_NT']),
                                        QC = data.loc[yearly_indices, 'QC'].unique()[0],
                                        method = methods
                                        )
                    MSE = dict(Year = year,
                                        RECO = mean_squared_error(data.loc[yearly_indices, 'RECO_DT'], data.loc[yearly_indices, 'RECO_NT'], squared=False),
                                        RECO_res = mean_squared_error(data.loc[yearly_indices, 'RECO_DT'], data.loc[yearly_indices, 'RECO_NT'], squared=False),
                                        GPP = mean_squared_error(data.loc[yearly_indices, 'GPP_DT'], data.loc[yearly_indices, 'GPP_NT'], squared=False),
                                        NEE = mean_squared_error(data.loc[yearly_indices, 'NEE_DT'], data.loc[yearly_indices, 'NEE_NT'], squared=False),
                                        QC = data.loc[yearly_indices, 'QC'].unique()[0],
                                        method = methods
                                        )
                        
                    results_R2 = results_R2.append(R2, ignore_index = True)
                    results_MSE = results_MSE.append(MSE, ignore_index = True)
                    

        condensed = dict()

        for methods in ['DT', 'NT', 'DTNT']:          
            condensed_method = pd.DataFrame(columns=['R2_med', 'R2_25', 'R2_75', 'MSE_med', 'MSE_25', 'MSE_75'])
            results_R2_method = results_R2.loc[results_R2['method']==methods,:]
            results_MSE_method = results_MSE.loc[results_MSE['method']==methods,:]
            condensed_method['R2_med'] = results_R2_method.quantile(0.5)
            condensed_method['R2_25'] = results_R2_method.quantile(0.25)
            condensed_method['R2_75'] = results_R2_method.quantile(0.75)
            condensed_method['MSE_med'] = results_MSE_method.quantile(0.5)
            condensed_method['MSE_25'] = results_MSE_method.quantile(0.25)
            condensed_method['MSE_75'] = results_MSE_method.quantile(0.75)
            condensed[methods] = condensed_method
            
        return results_R2, results_MSE, condensed

def condense(results_R2, results_MSE, syn=True, QC=True, RMSE=True):
    if QC:
        if RMSE:
            results_R2 = results_R2.loc[results_R2['QC']==0,:]
            results_MSE = results_MSE.loc[results_MSE['QC']==0,:]
            for key in ["RECO", "RECO_res", "GPP", "NEE", "NEE_clean"]:
                if key in results_MSE.keys():
                    results_MSE[key] = np.sqrt(results_MSE[key])
            
        else:
            results_R2 = results_R2.loc[results_R2['QC']==0,:]
            results_MSE = results_MSE.loc[results_MSE['QC']==0,:]
    else:
        if RMSE:
            for key in ["RECO", "RECO_res", "GPP", "NEE", "NEE_clean"]:
                if key in results_MSE.keys():
                    results_MSE[key] = np.sqrt(results_MSE[key])
            

    if RMSE:
        suffix = 'RMSE'
    else:
        suffix = "MSE"

    if syn:
        if RMSE:
            condensed = pd.DataFrame(columns=['R2_med', 'R2_25', 'R2_75', 'RMSE_med', 'RMSE_25', 'RMSE_75'])
        else:
            condensed = pd.DataFrame(columns=['R2_med', 'R2_25', 'R2_75', 'MSE_med', 'MSE_25', 'MSE_75'])
        condensed['R2_med'] = results_R2.quantile(0.5)
        condensed['R2_25'] = results_R2.quantile(0.25)
        condensed['R2_75'] = results_R2.quantile(0.75)
        condensed[suffix +'_med'] = results_MSE.quantile(0.5)
        condensed[suffix +'_25'] = results_MSE.quantile(0.25)
        condensed[suffix +'_75'] = results_MSE.quantile(0.75)
        try:
            condensed = condensed.drop(['Year', 'QC'])
        except:
            pass
    else:
        if RMSE:
            condensed = pd.DataFrame(columns=['R2_med', 'R2_25', 'R2_75', 'RMSE_med', 'RMSE_25', 'RMSE_75'])
        else:
            condensed = pd.DataFrame(columns=['R2_med', 'R2_25', 'R2_75', 'MSE_med', 'MSE_25', 'MSE_75'])        
        for method in ['DT', 'NT', 'DTNT']:
            if RMSE:
                condensed_method = pd.DataFrame(columns=['R2_med', 'R2_25', 'R2_75', 'RMSE_med', 'RMSE_25', 'RMSE_75', 'method'])     
            else:
                condensed_method = pd.DataFrame(columns=['R2_med', 'R2_25', 'R2_75', 'MSE_med', 'MSE_25', 'MSE_75', 'method'])     
            results_R2_method = results_R2.loc[results_R2['method']==method,:]
            results_MSE_method = results_MSE.loc[results_MSE['method']==method,:]
            condensed_method['R2_med'] = results_R2_method.quantile(0.5)
            condensed_method['R2_25'] = results_R2_method.quantile(0.25)
            condensed_method['R2_75'] = results_R2_method.quantile(0.75)
            condensed_method[suffix +'_med'] = results_MSE_method.quantile(0.5)
            condensed_method[suffix + '_25'] = results_MSE_method.quantile(0.25)
            condensed_method[suffix +'_75'] = results_MSE_method.quantile(0.75)
            condensed_method['method'] = method
            condensed = pd.concat([condensed, condensed_method])
        try:
            condensed = condensed.drop(['Year', 'QC'])
        except:
            pass
    return condensed

def full_analysis(csv_path, visualization=True):
    chunksize = 10000

    # the list that contains all the dataframes
    list_of_dataframes = []

    for df in pd.read_csv('/home/kaicohrs/Repositories/results/DML_sota/analysis_data.csv', chunksize=chunksize):
        # process your data frame here
        # then add the current data frame into the list
        #df = df.loc[(df['site']=='US-SRG'),:]
        #df = df.loc[np.isin(df['site'],only_one_year),:]
        #df = df.loc[(df['quality_mask']!=0),:]
        #df = df.loc[(df['MeasurementNEE_mask']!=0),:]
        df = df[['doy', 'Year', 'RECO_ann', 'RECO_DT', 'quality_mask',
                'Month', 'site', 'RECO_orth', 'NEE_orth', 'Time', 'NEE_ann', 'GPP_ann', 'GPP_DT', 
                'MeasurementNEE_mask', 'NEE',
                'QC', 'NIGHT', 'NEE_DT', 'RECO_NT', 'GPP_orth', 'RECO_orth_res',
                'GPP_NT', 'NEE_NT',]]
        list_of_dataframes.append(df)

    # if you want all the dataframes together, here it is
    data = pd.concat(list_of_dataframes)
    
    daily_avg = data.groupby(['site', 'Year', 'doy'])['quality_mask'].mean()
    daily_avg = daily_avg.rename('quality_mask_daily')
    data = data.join(daily_avg, on=['site', 'Year', 'doy'])
    data['quality_mask_daily'] = data['quality_mask_daily'].apply(lambda x: 0 if x < 0.5 else 1)
    
    data = data.loc[(data['quality_mask']!=0),:]
    data = data.loc[(data['QC']==0),:]
    data = data.loc[(data['MeasurementNEE_mask']!=0),:]
    data = data.loc[(~np.isnan(data['RECO_ann'])),:]
    data = data.loc[(data['quality_mask_daily']!=0),:]
    
    for flux in ["GPP", "RECO"]:
        for method in ["ann", "orth"]:
            results_all = list()

            flux = "RECO"
            method_1 = "NT"
            method_2 = "NT"

            for site in data["site"].unique():
                for year in data.loc[(data["site"] == site),'Year'].unique():
                    site_year = data.loc[(data["site"] == site) & (data['Year'] == year),:]
                    results_all.append(mean_squared_error(site_year[flux + "_" + method_1], site_year[flux + "_" + method_2], squared=False))

            np.median(results_all), np.quantile(results_all, 0.25), np.quantile(results_all, 0.75) 