import os
from datetime import date
import itertools
from os import listdir
import json
from re import L
import pathlib

import numpy as np
import pandas as pd

import shutil
import dml4fluxes.datasets.relevant_variables as relevant_variables

#sys.path.append('../')
from dml4fluxes.analysis.visualization import mean_diurnal_cycle, seasonal_cycle_plot,\
                                                cross_consistency_plots
from dml4fluxes.datasets.preprocessing import load_data, unwrap_time, standardize_column_names,\
                                                sw_pot_sm, sw_pot_sm_diff,\
                                                diffuse_to_direct_rad, NEE_quality_masks,\
                                                quality_check, GPP_prox,\
                                                normalize, wdefcum, check_available_variables,\
                                                make_cyclic, sw_pot_diff
from dml4fluxes.datasets.generate_data import synthetic_dataset
                                                
from dml4fluxes.analysis.postprocessing import evaluate, timely_averages, condense
from dml4fluxes.models import models
from .utility import get_available_sites, get_igbp_of_site, transform_t, JSONEncoder

class FluxPartDML():
    
    def __init__(self):
        self.PATH = None
        self.experiment_config = None
        self.dataset_config = None
        self.model_configs = None
        
    def new(self, site):
        #Start a new experiment.
        self.date = date.today().strftime("%Y-%m-%d")
        self.experiment_type = "FPDML"
        self.site = site

        path = pathlib.Path(__file__).parent.parent.parent.joinpath('results')
        self.experiment_name = f'{self.experiment_type}_{self.site}_{self.date}'

        i = 1
        while f'{self.experiment_name}_{i}' in listdir(path):
            i += 1
        self.experiment_name = f'{self.experiment_name}_{i}'
        self.PATH = path.joinpath(f'{self.experiment_name}')
        
        os.mkdir(self.PATH)
        print("Create new experiment folder : {}".format(self.PATH)) 
        
        
    def configs(self, experiment_config, dataset_config, model_configs):
        #Save and store the config files
        self.experiment_config = experiment_config
        self.dataset_config = dataset_config
        self.model_configs = model_configs
        
        with open(self.PATH.joinpath('experiment_config.txt'), 'w') as outfile:
            json.dump(self.experiment_config, outfile)    
        with open(self.PATH.joinpath('dataset_config.txt'), 'w') as outfile:
            json.dump(self.dataset_config, outfile)    
        with open(self.PATH.joinpath('model_configs.txt'), 'w') as outfile:
            json.dump(self.model_configs, outfile)    

    def prepare_data(self):
        #Run a data preprocessing pipeline
        self.data, self.data_path = load_data(self.dataset_config['site_name'], 
                                            year=2015, add_ann=~self.dataset_config['syn'], 
                                            files_path=True)
        self.data = unwrap_time(self.data)
        self.data = self.data.set_index('DateTime')
        self.data['DateTime'] = self.data.index
        self.data = standardize_column_names(self.data)
        self.data = self.data[list(set(self.data.columns)
                                & set(relevant_variables.variables))]
        self.data = NEE_quality_masks(self.data)
        
        self.data['SW_POT_sm'] = sw_pot_sm(self.data)
        self.data['SW_POT_sm_diff'] = sw_pot_sm_diff(self.data)
        self.data['SW_POT_diff'] = sw_pot_diff(self.data)
        self.data['CWD'] = wdefcum(self.data)
        self.data['SW_ratio'] = diffuse_to_direct_rad(self.data)
        self.data["doy"] = pd.to_datetime(self.data['DateTime']).dt.dayofyear
        self.data["tod"] = pd.to_datetime(self.data['DateTime']).dt.hour*60 + pd.to_datetime(self.data['DateTime']).dt.minute
        self.data["doy_sin"], self.data["doy_cos"] = make_cyclic(self.data["doy"])
        self.data["tod_sin"], self.data["tod_cos"] = make_cyclic(self.data["tod"])
        
        self.data['GPP_prox']= GPP_prox(self.data)
        #self.data['WD_sin'], data['WD_cos'] = WD_trans(self.data)
        self.data['NEE_NT'] = -self.data['GPP_NT'] + self.data['RECO_NT']
        self.data['NEE_DT'] = -self.data['GPP_DT'] + self.data['RECO_DT']
        
        # Setting another GT flux (could be DT, NT or NEE for instance)
        if self.dataset_config['alternative_fluxes']:
            self.data['NEE'] = self.data['NEE_' + self.dataset_config['alternative_fluxes']]
            self.data['RECO'] = self.data['RECO_' + self.dataset_config['alternative_fluxes']]
            self.data['GPP'] = self.data['GPP_' + self.dataset_config['alternative_fluxes']]
        if self.dataset_config['alternative_treatment']:
            self.data['SW_IN'] = self.data[self.dataset_config['alternative_treatment']]
        
        # Reduce variables to the ones also available
        self.W_var = check_available_variables(self.dataset_config['W'], self.data.columns)
        self.X_var = check_available_variables(self.dataset_config['X'], self.data.columns)

        # Generate mask for quality of all data to be used.
        self.data['QC'] = quality_check(self.data, self.X_var + self.W_var + ['SW_IN'])
        
        # Drop problematic rows
        self.data = self.data.replace(-9999, np.nan)
        
        # TODO: data generation happens in a different script. We only load similar to the other
        # TODO: data from fluxnet and join the previously generated data
        if self.dataset_config['syn']:
            self.data = synthetic_dataset(data=self.data,
                                        Q10=self.dataset_config['Q10'],
                                        relnoise=self.dataset_config['relnoise'],
                                        version=self.dataset_config['version'],
                                        pre_computed=self.dataset_config['pre_computed'],
                                        site_name=self.dataset_config['site_name'])
            self.data = self.data[~pd.isna(self.data[['NEE_syn'] + ['NEE_syn_clean']]).any(axis=1)]
        
        # Normalize/Standardize Variables including for testing (always the next year)
        # Or previous year for the last one.
        for var in self.X_var + self.W_var + ['SW_IN']:
            if var.endswith('_n') or var.endswith('_s'):
                self.data = normalize(self.data, var[:-2], norm_type=var[-1])  
            else:
                pass
            
        self.data = self.data[~pd.isna(self.data[self.X_var + self.W_var + ['SW_IN'] + ['NEE']]).any(axis=1)]
        #self.data = self.data[~pd.isna(self.data[self.X_var + self.W_var]).any(axis=1)]


    def fit_models(self):
        # Run the fitting and partitioning
        self.fitted_models = dict()
        if self.dataset_config['syn']:
            self.data['NEE_QC'] = 0
            #TODO: Not sure if relevant
            self.data['QC']=0

        self.data['T'] = self.data['SW_IN']
        self.data['GPP_orth'] = 0
        self.data['RECO_orth'] = 0
        self.data['NEE_orth'] = 0
        self.data['T_orth'] = 0
        self.data['woy'] = -99
        
        if self.dataset_config['years'] == 'all':
            years = self.data['Year'].unique()
        else:
            years = self.dataset_config['years']
            #self.data = self.data[self.data['Year'].isin(self.dataset_config['years'])]
            
        for year in years:
            print(year)
            indices = (self.data['Year']==year)
            test_indices = (self.data['Year']==year+1)
            
            # Use only measured data for training
            if self.dataset_config['good_years_only']:
                if self.data[indices].QC.unique() > 0:
                    self.data.loc[indices, 'GPP_orth'] = None
                    self.data.loc[indices, 'RECO_orth'] = None
                    self.data.loc[indices, 'RECO_orth_res'] = None
                    self.data.loc[indices, 'NEE_orth'] =  None
                    self.data.loc[indices, 'T_orth'] = None
                    self.data.loc[indices, 'LUE_orth'] = None
                    print(f"{year} did not pass quality check and is disregarded.")
                    continue
                
            mask = (self.data['Year']==year) & (self.data['NEE_QC']==0)
            mask_test = (self.data['Year']==year+1) & (self.data['NEE_QC']==0)
            
            #TODO: Potentially filter out years that do not fulfill criterion
            #self.data = self.data[~pd.isna(self.data[self.X_var + self.W_var + ['SW_IN'] + ['NEE']]).any(axis=1)]
            if self.dataset_config['transform_T']:
                transform_mask = (self.data['Year']==year)
                #transform_mask = mask
            
                if self.dataset_config['syn']:
                    target = 'NEE_syn'
                else:
                    target = 'NEE'

                self.data.loc[transform_mask, 'T'], parameter = transform_t(x=self.data.loc[transform_mask, 'SW_IN'],
                                                            delta=self.dataset_config['delta'],
                                                            data=self.data.loc[transform_mask,:],
                                                            month_wise=self.dataset_config['month_wise'],
                                                            moving_window=self.dataset_config['moving_window'],
                                                            target = target,
                                                            )
                self.parameter = parameter
                
                # self.T_test, _, _ = transform_t(x=self.data.loc[mask_test, 'SW_IN'],
                #                             delta=self.dataset_config['delta'],
                #                             data=self.data.loc[mask_test,:],
                #                             month_wise=self.dataset_config['month_wise'],
                #                             moving_window=self.dataset_config['moving_window'],
                #                             parameter = parameter)
            
            #self.T = self.data.loc[mask, 'GPP_DT']
            self.T = self.data.loc[mask, 'T'].values
            
            self.X = self.data.loc[mask, self.X_var].values
            self.X_test = self.data.loc[mask_test, self.X_var].values
            
            if not len(self.W_var):
                self.W = None
                self.W_test = None
            else:
                self.W = self.data.loc[mask, self.W_var].values
                self.W_test = self.data.loc[mask_test, self.W_var].values
                
            if self.dataset_config['syn']:
                self.Y = self.data.loc[mask, 'NEE_syn'].values
            else:
                self.Y = self.data.loc[mask, 'NEE'].values
                self.Y_test = self.data.loc[mask_test, 'NEE'].values
            

            self.dml = models.dml_fluxes(self.model_configs['y'],
                                        self.model_configs['t'],
                                        self.model_configs['final'],
                                        self.model_configs['dml'])
            self.dml.fit(self.Y,self.X,self.T,self.W)
            
            if self.dataset_config['syn']:
                self.dml.score_train = self.dml.get_score(self.X, self.T, 
                                                        self.data.loc[mask, 'NEE_syn_clean'], self.W)
            else:
                self.dml.score_train = self.dml.get_score(self.X, self.T, self.Y, self.W)
            #self.dml.score_test = self.dml.get_score(self.X_test, self.T_test, self.Y_test, self.W_test)

            if self.dataset_config['transform_T']:
                self.light_response_curve = models.LightResponseCurve(self.dataset_config['moving_window'],
                                                                        self.parameter, 
                                                                        self.dataset_config['delta'], 
                                                                        self.dml.lue)


            self.fitted_models[str(year)] = self.dml

            # Take all data for that year to obtain the partitioning            
            if self.dataset_config['transform_T']:
                # Since we provide the parameter the function does not fit but only transforms
                
                self.data.loc[indices, 'T'], alphas, betas = transform_t(x=self.data.loc[indices, 'SW_IN'],
                                                            delta=self.dataset_config['delta'],
                                                            data=self.data.loc[indices,:],
                                                            month_wise=self.dataset_config['month_wise'],
                                                            moving_window=self.dataset_config['moving_window'],
                                                            parameter = parameter)
                
            #T = self.data.loc[indices, 'GPP_DT']
            T = self.data.loc[indices, 'T'].values
            X = self.data.loc[indices, self.X_var].values
        
            if self.dataset_config['transform_T']:            
                self.data.loc[indices, "alpha"] = alphas
                self.data.loc[indices, "beta"] = betas
            
            if not len(self.W_var):
                W = None
            else:
                W = self.data.loc[indices, self.W_var].values

            if self.dataset_config['syn']:
                Y = self.data.loc[indices, 'NEE_syn'].values
            else:
                Y = self.data.loc[indices, 'NEE'].values

            # Run predictions
            self.data.loc[indices, 'GPP_orth'] = self.dml.gpp(X, T)
            self.data.loc[indices, 'RECO_orth'] = self.dml.reco(X, T, W)
            if 'RECO_orth_res' not in self.data.columns:
                self.data['RECO_orth_res'] = None
            self.data.loc[indices, 'RECO_orth_res'] = self.dml.reco_res(Y, X, T)
            self.data.loc[indices, 'NEE_orth'] = self.dml.nee(X, T, W)
            self.data.loc[indices, 'T_orth'] = self.dml.t(X, W)
            if 'LUE_orth' not in self.data.columns:
                self.data['LUE_orth'] = None
            self.data.loc[indices, 'LUE_orth'] = self.dml.lue(X)
        
        folder_name = self.data_path.name
        folder_path = self.data_path.parent.joinpath('DMLPartitioning',
                                        self.experiment_name, folder_name)
        folder_path.mkdir(parents=True, exist_ok=False)
        if self.dataset_config['transform_T']:
            params = ['alpha', 'beta']
        else:
            params = []
            
        self.data[['GPP_orth', 'RECO_orth', 'RECO_orth_res', 'NEE_orth', 'T_orth',
                'LUE_orth', 'T'] + params].to_csv(folder_path.joinpath('orth_partitioning.csv'))

        return None
    
    def all_partitions(self, year=2015):
        #Run the partitioning pipeline over all available sites
        path = os.path.join(os.path.dirname(__file__), f"../../data/Fluxnet-{year}")
        
        if self.dataset_config['test_scenario']:
            sites = ['AU-Cpr', 'DE-Gri', 'BE-Lon',
                    'FI-Hyy', 'DK-Sor', 'GF-Guy']
        else:
            sites = get_available_sites(path)

        progress = pd.DataFrame(columns=['site', 'status'])
        progress.to_csv(self.PATH.joinpath('progress.csv'))
        progress = pd.read_csv(self.PATH.joinpath('progress.csv'), index_col=0)
        #sites = ['AU-DaP', 'DE-Geb', 'CA-Qfo']

        for site in sites:
            print(f'Starting with site {site}.')
            self.dataset_config['site_name'] = site
            self.prepare_data()
            self.fit_models()
            status = "success"
            #print(f'Site {site} did not work for some reason.')
            #status = "failure"
            row = {'site':site, 'status':status}
            new_df = pd.DataFrame([row])
            progress = pd.concat([progress, new_df], axis=0, ignore_index=True)       
            progress.to_csv(self.PATH.joinpath('progress.csv'))
    
    
    def all_analysis(self, year=2015):
        #Run an analysis pipeline over all available sites
        path = os.path.join(os.path.dirname(__file__), f"../../data/Fluxnet-{year}")
        sites = get_available_sites(path)
        
        if self.dataset_config['test_scenario']:
            sites = ['FI-Hyy', 'DE-Gri', 'IT-MBo']        
        
        if self.dataset_config['years'] == 'all':
            self.all_years = True
        else:
            self.all_years = False
            
        if self.dataset_config['years'] == 'first':
            self.first_years = True
        else:
            self.first_years = False
            
        if self.dataset_config['years'] == 'last':
            self.last_years = True
        else:
            self.last_years = False
        
        self.results=dict()
        for key, part, method in list(itertools.product(["R2_", "MSE_"], ["all_", "day_", "night_"], ["orth", "ann"])):
            self.results[key + part + method] = pd.DataFrame()
        if self.experiment_config['extrapolate']:
            self.results_test=dict()
            for key, part, method in list(itertools.product(["R2_", "MSE_"], ["all_", "day_", "night_"], ["orth", "ann"])):
                self.results_test[key + part + method] = pd.DataFrame()
        
        progress = pd.DataFrame(columns=['site', 'status'])
        progress.to_csv(self.PATH + '/progress.csv')
        progress = pd.read_csv(self.PATH + '/progress.csv', index_col=0)
        
        for site in sites:
            print(f'Starting with site {site}.')
            save = [
                'Time',
                'Month', 
                'Year', 
                'doy', 
                'NIGHT', 
                'site', 
                'code',
                #'WS', 
                # 'WD',
                'wdefcum', 
                'wdefcum_n',
                'SW_ratio', 
                'SW_POT', 
                'SW_POT_sm', 
                'SW_POT_sm_diff', 
                #'T_orth', 
                'LUE_orth', 
                'TA', 
                'TA_n',
                'SW_IN', 
                'VPD', 
                'VPD_n',
                'QC',
                # 'PA', 
                # 'P', 
                # 'WS', 
                #'LE', 
                # 'SWC_1', 
                # 'SWC_2', 
                # 'TS_1', 
                # 'TS_2',
                'quality_mask',
                'MeasurementNEE_mask',
                'NEE_syn', 
                'NEE_syn_clean', 
                'RECO_syn', 
                'GPP_syn',
                'NEE', 
                'NEE_orth', 
                'GPP_DT', 
                'NEE_DT', 
                'NEE_NT',
                'GPP_NT', 
                'GPP_orth', 
                'RECO_DT', 
                'RECO_NT',
                'RECO_ann', 
                'GPP_ann', 
                'NEE_ann',
                'RECO_orth', 
                'RECO_orth_res',
                'Rb_res',
                'Rb',
                'Q10',
                'Q10_res'
            ]
            
            self.dataset_config['site_name'] = site
            self.prepare_data()
            self.fit_models()
            
            self.data['site'] = site
            try:
                self.data['code'] = get_igbp_of_site(site)
            except:
                print(f'IGBP code of {site} not found. Set to "unknown"')

            save = list(set(save) & set(self.data.columns))

            if "analysis_data.csv" not in listdir(self.PATH):
                self.data[save].to_csv(self.PATH + '/analysis_data.csv')
                analysis_data = pd.read_csv(self.PATH + '/analysis_data.csv')
                try:
                    analysis_data = analysis_data.drop('Unnamed: 0', axis=1)
                except:
                    pass  
            else:
                analysis_data = pd.read_csv(self.PATH + '/analysis_data.csv')
                try:
                    analysis_data = analysis_data.drop('Unnamed: 0', axis=1)
                except:
                    pass  
                analysis_data = pd.concat([analysis_data,self.data[save]],ignore_index=True)
                analysis_data.to_csv(self.PATH + '/analysis_data.csv')

        
            if self.dataset_config['syn']:
                methods = ['orth']
            else:
                methods = ['orth', 'ann']

            for method in methods:
                for part in ['all', 'day', 'night']:
                    results_R2, results_MSE, condensed = evaluate(self.data.loc[(self.data['quality_mask']==1) & (self.data['QC']==0),:], self.dataset_config['syn'], False, part=part, method=method)
                    results_R2['site'] = site
                    results_MSE['site'] = site
                    
                    self.results["R2_"+part+"_"+method] = pd.concat([self.results["R2_"+part+"_"+method], results_R2])
                    self.results["MSE_"+part+"_"+method] = pd.concat([self.results["MSE_"+part+"_"+method], results_MSE])
            
            if self.experiment_config['extrapolate']:
                self.data_test['site'] = site
                try:
                    self.data_test['code'] = get_igbp_of_site(site)
                except:
                    print(f'IGBP code of {site} not found. Set to "unknown"')

                save = list(set(save) & set(self.data_test.columns))

                if "analysis_data_test.csv" not in listdir(self.PATH):
                    self.data_test[save].to_csv(self.PATH + '/analysis_data_test.csv')
                else:
                    analysis_data_test = pd.read_csv(self.PATH + '/analysis_data_test.csv')
                    try:
                        analysis_data_test = analysis_data_test.drop('Unnamed: 0', axis=1)
                    except:
                        pass  
                    analysis_data_test = pd.concat([analysis_data,self.data_test[save]],ignore_index=True)
                    analysis_data_test.to_csv(self.PATH + '/analysis_data_test.csv')
                
                            
                for part in ['all', 'day', 'night']:
                    results_R2, results_MSE, condensed = evaluate(self.data_test, self.dataset_config['syn'], False, part=part)
                    results_R2['site'] = site
                    results_MSE['site'] = site
                    
                    self.results_test["R2_"+part] = pd.concat([self.results_test["R2_"+part], results_R2])
                    self.results_test["MSE_"+part] = pd.concat([self.results_test["MSE_"+part], results_MSE])
                
            print(f"Finished site {site}")
            progress = progress.append({'site':site, 'status':'success'}, ignore_index=True)
            progress.to_csv(self.PATH + '/progress.csv')
            #except:
            #    print(f"Failed site {site}")
            #    progress = progress.append({'site':site, 'status':'fail'}, ignore_index=True)
            #    progress.to_csv(self.PATH + '/progress.csv')
            #print(f"Something went wrong with site {site}")
                    
            if self.all_years:
                self.dataset_config['years'] = 'all'
            elif self.first_years:
                self.dataset_config['years'] = 'first'
            elif self.last_years:
                self.dataset_config['years'] = 'last'

        with open(self.PATH + '/results_all.json', 'w') as fp:
            json.dump(self.results, fp, cls=JSONEncoder)
            #json.load(open('result.json')
            #pd.read_json(json.load(open('result.json'))['1'])
        if self.experiment_config['extrapolate']:  
            with open(self.PATH + '/results_all_test.json', 'w') as fp:
                json.dump(self.results_test, fp, cls=JSONEncoder)

        for method in methods:   
            for part in ['all', 'day', 'night']:
                condensed = condense(self.results["R2_"+part+"_"+method], self.results["MSE_"+part+"_"+method], syn=self.dataset_config['syn'], QC=True, RMSE=self.dataset_config['RMSE'])
                condensed.to_csv(self.PATH + "/results_" + part + "_"+ method + ".csv")
        if self.experiment_config['extrapolate']:  
            condensed = condense(self.results_test["R2_all"], self.results_test["MSE_all"], syn=self.dataset_config['syn'], QC=True, RMSE=self.dataset_config['RMSE'])
            condensed.to_csv(self.PATH + "/results_all_test.csv")
        
        
        analysis_data = pd.read_csv(self.PATH + '/analysis_data.csv')
        try:
            analysis_data = analysis_data.drop('Unnamed: 0', axis=1)
        except:
            pass 
        
        analysis_data = get_average_quality_mask(analysis_data)
        analysis_data = analysis_data.loc[(analysis_data['quality_mask']==1) & (analysis_data['QC']==0),:]
        analysis_data= timely_averages(analysis_data)
        
        southern_hemisphere = ["AU-Cpr",
                                "AU-DaP",
                                "AU-Dry",
                                "AU-How",
                                "AU-Stp",
                                "ZA-Kru",
                                "GF-Guy"] #Northern hemisphere but still filtered out due to lat < 15
        
        image_folder = self.PATH + '/images'
        os.mkdir(image_folder)
        for method in methods:
            cross_consistency_plots(analysis_data, image_folder, method=method)
        
        analysis_data = analysis_data.loc[~np.isin(analysis_data['site'],southern_hemisphere),:]
        for method in methods:
            seasonal_cycle_plot(analysis_data, image_folder, method=method)
        
        only_one_year = ["IT-Cpz",
                            "Be-Lon",
                            "IT-Ro1",
                            "US-ARM",
                            "US-MMS", # For some weird reason hourly data
                            "US-UMB"] # For some weird reason hourly data
        
        analysis_data = analysis_data.loc[~np.isin(analysis_data['site'],only_one_year),:]
        for method in methods:
            mean_diurnal_cycle(analysis_data, 'GPP', image_folder, method=method)
            mean_diurnal_cycle(analysis_data, 'RECO', image_folder, method=method)
        
        if self.experiment_config['extrapolate']:
            image_folder = self.PATH + '/images_test'
            os.mkdir(image_folder) 
            analysis_data_test=timely_averages(analysis_data_test)
            
            cross_consistency_plots(analysis_data_test, image_folder)
            seasonal_cycle_plot(analysis_data_test, image_folder)
            analysis_data_test = pd.read_csv(self.PATH + '/analysis_data_test.csv')
            mean_diurnal_cycle(analysis_data_test, 'GPP', image_folder)
            mean_diurnal_cycle(analysis_data_test, 'RECO', image_folder)
            
    def rm_experiment(self):
        shutil.rmtree(self.PATH)