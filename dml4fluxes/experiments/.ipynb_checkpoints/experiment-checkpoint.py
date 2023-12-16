import torch
from datetime import date
import itertools

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LinearRegression

import random
import pandas as pd

from os import listdir
import json

import doubleml as dml
from econml.dml import DML, LinearDML, SparseLinearDML, NonParamDML, CausalForestDML, KernelDML
from econml.utilities import WeightedModelWrapper
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import itertools

#sys.path.append('../')
from bayesian_q10.analysis.visualization import *
from bayesian_q10.datasets.preprocessing import *
from bayesian_q10.analysis.postprocessing import *
from bayesian_q10.models import models
from .utility import *

class FluxPartDML2():

    def __init__(self):
        self.PATH = None
        self.experiment_config = None
        self.dataset_config = None
        self.model_configs = None
        
    def new(self, site):
        self.date = date.today().strftime('%Y-%m-%d')
        self.experiment_type = 'FPDML'
        self.site = site
        
        path = '/usr/users/kcohrs/bayesian-q10/exp/'
        self.experiment_name = self.experiment_type + '_' + self.site + '_' + self.date       
        i = 1
        while True:
            self.number = "_{}".format(i)
            try:
                os.mkdir(path + self.experiment_name + self.number)
                break
            except:
                i += 1
        self.experiment_name = self.experiment_name + self.number
        self.PATH = path + self.experiment_name
        print("Create new experiment folder : {}".format(self.PATH)) 
        
    def configs(self, experiment_config, dataset_config, model_configs):

        self.experiment_config = experiment_config
        self.dataset_config = dataset_config
        self.model_configs = model_configs
        
        with open(self.PATH + '/experiment_config.txt', 'w') as outfile:
            json.dump(self.experiment_config, outfile)    
        with open(self.PATH + '/dataset_config.txt', 'w') as outfile:
            json.dump(self.dataset_config, outfile)    
        with open(self.PATH + '/model_configs.txt', 'w') as outfile:
            json.dump(self.model_configs, outfile)    

    def prepare_data(self):
        self.data = load_data(self.dataset_config['site_name'])
        self.data = unwrap_time(self.data)
        self.data = standardize_column_names(self.data)
        self.data = sw_pot_sm(self.data)
        self.data = sw_pot_sm_diff(self.data)
        self.data = w_def_cum(self.data)
        self.data = diffuse_to_direct_rad(self.data)
        self.data = GPP_prox(self.data)
        
        self.W_var = list(set(self.dataset_config['W']) & set(self.data.columns))
        if len(list(set(self.dataset_config['W']) - (set(self.W_var)))):
            print(f'For site {self.dataset_config["site_name"]} variables {list(set(self.dataset_config["W"]) - set(self.W_var))} in W not available')

        self.X_var = list(set(self.dataset_config['X']) & set(self.data.columns))
        if len(list(set(self.dataset_config['X']) - (set(self.X_var)))):
            print(f'For site {self.dataset_config["site_name"]} variables {list(set(self.dataset_config["X"]) - set(self.X_var))} in X not available')
        
        # drop problematic rows
        self.data.replace(-9999, np.nan)
        self.data = self.data[~pd.isna(self.data[self.X_var + self.W_var + ['SW_IN']]).any(axis=1)]
        
        if self.dataset_config['syn']:
            self.data = synthetic_dataset(self.data, self.dataset_config['Q10'], self.dataset_config['relnoise'])
        self.data = normalize(self.data, self.X_var + self.W_var, self.dataset_config['norm_type'])
        if self.experiment_config['extrapolate']:
            self.data_test = self.data.copy()
            self.data_test = normalize_test(self.data_test, self.X_var + self.W_var, self.dataset_config['norm_type'])
    
    def fit_models(self):
        self.fitted_models = dict()
        self.data = quality_check(self.data, self.X_var + self.W_var + ['SW_IN'])
        self.data['T'] = self.data['SW_IN']
        self.data['GPP_orth'] = 0
        self.data['RECO_orth'] = 0
        self.data['RECO_orth_res'] = 0
        self.data['NEE_orth'] = 0
        self.data['T_orth'] = 0
        self.all_years = False
        if self.experiment_config['extrapolate']:
            self.data_test = quality_check(self.data_test, self.X_var + self.W_var + ['SW_IN'])
            self.data_test['T'] = self.data_test['SW_IN']

        
        #if True:
        #    self.data = self.data.drop(self.data[self.data['Year'] == self.data['Year'].unique().max()].index)
        
        if self.dataset_config['years'] == 'all':
            self.all_years = True
            self.dataset_config['years'] = self.data['Year'].unique()
        else:
            self.data = self.data[self.data['Year'].isin(self.dataset_config['years'])]
            if self.experiment_config['extrapolate']:
                self.data_test = self.data_test[self.data_test['Year'].isin(self.dataset_config['years'])]                
        
        for year in self.dataset_config['years']:
            print(year)
            indices = (self.data['Year']==year)

            if self.dataset_config['transform_T']:
                self.data.loc[indices, 'T'], parameter = transform_T(x=self.data.loc[indices, 'SW_IN'], 
                                                            delta=self.dataset_config['delta'], 
                                                            data=self.data.loc[indices,:],
                                                            month_wise=self.dataset_config['month_wise']
                                                            )    
            T = self.data.loc[indices, 'T']            
            X = self.data.loc[indices, [var + '_n' for var in self.X_var]]            
    
            if not len(self.W_var):
                W = None
            else:
                W = self.data.loc[indices, [var + '_n' for var in self.W_var]]
            
            if self.dataset_config['syn']:
                Y = self.data.loc[indices, 'NEE_syn']
            else:
                Y = self.data.loc[indices, 'NEE']
                    
                        
            self.dml = models.dml_fluxes(self.model_configs['y'], self.model_configs['t'], self.model_configs['final'], self.model_configs['dml'])
            self.dml.fit(Y,X,T,W)

            self.fitted_models[str(year)] = self.dml
    
            # Run predictions
            self.data.loc[indices, 'GPP_orth'] = self.dml.gpp(X, T)
            self.data.loc[indices, 'RECO_orth'] = self.dml.reco(X, T, W)
            self.data.loc[indices, 'RECO_orth_res'] = self.dml.reco_res(Y, X, T)
            self.data.loc[indices, 'NEE_orth'] = self.dml.nee(X, T, W)
            self.data.loc[indices, 'T_orth'] = self.dml.t(X, W)
            self.data.loc[indices, 'LUE_orth'] = self.dml.lue(X)
            
            
            # Do extrapolation to the consecutive year
            if self.experiment_config['extrapolate'] and year != max(self.dataset_config['years']):
                indices = (self.data_test['Year']==year+1)
                if self.dataset_config['transform_T']:
                    self.data_test.loc[indices, 'T'] = transform_T(x=self.data_test.loc[indices, 'SW_IN'], 
                                                                delta=self.dataset_config['delta'], 
                                                                data=self.data_test.loc[indices,:],
                                                                month_wise=self.dataset_config['month_wise'],
                                                                parameter = parameter)
                T = self.data_test.loc[indices, 'T']
                X = self.data_test.loc[indices, [var + '_n' for var in self.X_var]]            
                if not len(self.W_var):
                    W = None
                else:
                    W = self.data_test.loc[indices, [var + '_n' for var in self.W_var]]

                if self.dataset_config['syn']:
                    Y = self.data_test.loc[indices, 'NEE_syn']
                else:
                    Y = self.data_test.loc[indices, 'NEE']
                # Run predictions
                self.data_test.loc[indices, 'GPP_orth'] = self.dml.gpp(X, T)
                self.data_test.loc[indices, 'RECO_orth'] = self.dml.reco(X, T, W)
                self.data_test.loc[indices, 'RECO_orth_res'] = self.dml.reco_res(Y, X, T)
                self.data_test.loc[indices, 'NEE_orth'] = self.dml.nee(X, T, W)
                self.data_test.loc[indices, 'T_orth'] = self.dml.t(X, W)
                self.data_test.loc[indices, 'LUE_orth'] = self.dml.lue(X)
        
        if self.experiment_config['extrapolate']:
            self.data_test = self.data_test.drop(self.data_test[(self.data_test['Year']==self.data_test['Year'].min())].index, axis=0)        
            
    def all_analysis(self):
        sites = get_available_sites("/usr/users/kcohrs/bayesian-q10/data")
        
        if self.dataset_config['years'] == 'all':
            self.all_years = True
        else:
            self.all_years = False
        
        self.results=dict()
        for key, part in list(itertools.product(["R2_", "MSE_"], ["all", "day", "night"])):
            self.results[key + part] = pd.DataFrame()
        if self.experiment_config['extrapolate']:
            self.results_test=dict()
            for key, part in list(itertools.product(["R2_", "MSE_"], ["all", "day", "night"])):
                self.results_test[key + part] = pd.DataFrame()
        
        for site in sites:
            print(f'Starting with site {site}.')
            save = ['Time','Month', 'Year', 'doy', 'NIGHT', 'site', 'code',
            #'WS', 'WD'
            'wdefcum', 'wdefcum_n',
            'SW_ratio', 
            'SW_POT', 
            'SW_POT_sm', 'SW_POT_sm_diff', 
            #'T_orth', 
            'LUE_orth', 
            'TA', "TA_n",
            'SW_IN', 'VPD', 
            'VPD_n',
            # 'PA', 'P', 'WS', 
            #'LE', 'SWC_1', 'SWC_2', 'TS_1', 'TS_2', 
            'NEE_syn', 'NEE_syn_clean', 'RECO_syn', 'GPP_syn', #for synthetic data
            'NEE', 'NEE_orth', 'GPP_DT', 
            'GPP_NT', 'GPP_orth', 'RECO_DT', 'RECO_NT', 
            'RECO_orth', 'RECO_orth_res']
            
            
            self.dataset_config['site_name'] = site
            self.prepare_data()
            self.fit_models()
            
            self.data['site'] = site
            try:
                self.data['code'] = get_IGBP_of_site(site)
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
            
            
            for part in ['all', 'day', 'night']:
                results_R2, results_MSE, condensed = evaluate(self.data, self.dataset_config['syn'], False, part=part)
                results_R2['site'] = site
                results_MSE['site'] = site
                
                self.results["R2_"+part] = pd.concat([self.results["R2_"+part], results_R2])
                self.results["MSE_"+part] = pd.concat([self.results["MSE_"+part], results_MSE])
            
            if self.experiment_config['extrapolate']:
                self.data_test['site'] = site
                try:
                    self.data_test['code'] = get_IGBP_of_site(site)
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
            #print(f"Something went wrong with site {site}")
                    
            if self.all_years:
                self.dataset_config['years'] = 'all'

        with open(self.PATH + '/results_all.json', 'w') as fp:
            json.dump(self.results, fp, cls=JSONEncoder)
            #json.load(open('result.json')
            #pd.read_json(json.load(open('result.json'))['1'])
        if self.experiment_config['extrapolate']:  
            with open(self.PATH + '/results_all_test.json', 'w') as fp:
                json.dump(self.results_test, fp, cls=JSONEncoder)

            
        for part in ['all', 'day', 'night']:
            condensed = condense(self.results["R2_"+part], self.results["MSE_"+part], syn=self.dataset_config['syn'], QC=True)
            condensed.to_csv(self.PATH + "/results_" + part + ".csv")
        if self.experiment_config['extrapolate']:  
            condensed = condense(self.results_test["R2_all"], self.results_test["MSE_all"], syn=self.dataset_config['syn'], QC=True)
            condensed.to_csv(self.PATH + "/results_all_test.csv")
            
        analysis_data=timely_averages(analysis_data)
        
        image_folder = self.PATH + '/images'
        os.mkdir(image_folder) 
        cross_consistency_plots(analysis_data, image_folder)
        seasonal_cycle_plot(analysis_data, image_folder)
        analysis_data = pd.read_csv(self.PATH + '/analysis_data.csv')
        mean_diurnal_cycle(analysis_data, 'GPP', image_folder)
        mean_diurnal_cycle(analysis_data, 'RECO', image_folder)
        
        if self.experiment_config['extrapolate']:
            image_folder = self.PATH + '/images_test'
            os.mkdir(image_folder) 
            analysis_data_test=timely_averages(analysis_data_test)
            
            cross_consistency_plots(analysis_data_test, image_folder)
            seasonal_cycle_plot(analysis_data_test, image_folder)
            analysis_data_test = pd.read_csv(self.PATH + '/analysis_data_test.csv')
            mean_diurnal_cycle(analysis_data_test, 'GPP', image_folder)
            mean_diurnal_cycle(analysis_data_test, 'RECO', image_folder)


    
class FluxPartDML():

    def __init__(self):
        self.PATH = None
        self.experiment_config = None
        
    def new(self, site):
        self.date = date.today().strftime('%Y-%m-%d')
        self.experiment_type = 'FPDML'
        self.site = site
        
        path = '/usr/users/kcohrs/bayesian-q10/exp/'
        self.experiment_name = self.experiment_type + '_' + self.site + '_' + self.date       
        i = 1
        while True:
            self.number = "_{}".format(i)
            try:
                os.mkdir(path + self.experiment_name + self.number)
                break
            except:
                i += 1
        self.experiment_name = self.experiment_name + self.number
        self.PATH = path + self.experiment_name
        print("Create new experiment folder : {}".format(self.PATH))   
        
    def configs(self, experiment_config):

        self.experiment_config = experiment_config
        
        with open(self.PATH + '/experiment_config.txt', 'w') as outfile:
            json.dump(self.experiment_config, outfile)    
    
    def get_estimators(self, training_type=None):

        if training_type != "monthly":
            def gpp(x, t):
                return -self.est.const_marginal_effect(x)*t
                
            def reco(x, t, w=None):
                return np.mean([self.est._models_nuisance[0][i]._model_y.predict(x, w)-self.est.const_marginal_effect(x)*(self.est._models_nuisance[0][i]._model_t.predict(x, w)) for i in range(len(self.est._models_nuisance[0]))], axis=0)

            def nee(x, t, w=None):
                return -gpp(x, t) + reco(x, t, w)
                
            def t(x, w=None):
                return np.mean([self.est._models_nuisance[0][i]._model_t.predict(x, w) for i in range(len(self.est._models_nuisance[0]))], axis=0)
            
        else:
            def gpp(x, t, months):
                out = list()
                for i, month in enumerate(months):
                    out.append(-self.est_dict[month].est.const_marginal_effect([x[i]])*t[i])
                return np.array(out)
                
            def reco(x, t, months, w=None):
                out = list()
                if w:
                    for j, month in enumerate(months):
                        out.append(np.mean([self.est_dict[month].est._models_nuisance[0][i]._model_y.predict(np.array([x[j]]), np.array([w[j]]))-self.est_dict[month].est.const_marginal_effect(np.array([x[j]]))*(self.est_dict[month].est._models_nuisance[0][i]._model_t.predict(np.array([x[j]]), np.array([w[j]]))) for i in range(len(self.est_dict[month].est._models_nuisance[0]))]))
                    return np.array(out)
                else:
                    for j, month in enumerate(months):
                        out.append(np.mean([self.est_dict[month].est._models_nuisance[0][i]._model_y.predict(np.array([x[j]]), W=None)-self.est_dict[month].est.const_marginal_effect(np.array([x[j]]))*(self.est_dict[month].est._models_nuisance[0][i]._model_t.predict(np.array([x[j]]), W=None)) for i in range(len(self.est_dict[month].est._models_nuisance[0]))]))
                    return np.array(out)
                
            def nee(x, t, months, w=None):
                return  -gpp(x, t, months).squeeze() + reco(x, t, months, w)
            
            def t(x, months, w=None):
                out = list()
                if w:
                    for j, month in enumerate(months):
                        out.append(np.mean([self.est_dict[month].est._models_nuisance[0][i]._model_t.predict(np.array([x[j]]), np.array([w[j]])) for i in range(len(self.est_dict[month].est._models_nuisance[0]))]))
                    return np.array(out)
                else:
                    for j, month in enumerate(months):
                        out.append(np.mean([self.est_dict[month].est._models_nuisance[0][i]._model_t.predict(np.array([x[j]]), W=None) for i in range(len(self.est_dict[month].est._models_nuisance[0]))]))
                    return np.array(out)
            
        return gpp, reco, nee, t

    
    def run(self):
        
        random.seed(self.experiment_config['seed'])
        np.random.seed(self.experiment_config['seed'])
        torch.manual_seed(self.experiment_config['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        try:
            os.mkdir(self.PATH + '/results')
        except:
            print('Results folder already exist. Experiments might have been run before.')
        
        
        ## Load and preprocess the data
        dataset = self.experiment_config['site']

        if dataset == 'syn':
            self.data_all = pd.read_csv('../data/Synthetic4BookChap.csv')
            if self.experiment_config['relnoise'] != 0:
                print(f'Attention: {self.experiment_config["relnoise"]} relnoise is added to NEE_syn')
                self.data_all['NEE_syn_clean'] = self.data_all['NEE_syn']
                self.data_all['NEE_syn'] = self.data_all['NEE_syn'] * (1 + self.experiment_config['relnoise'] * np.random.normal(size=self.data_all['NEE_syn'].size))
        else: 
            if not self.experiment_config['syn_data']:
                folder_path = '../data/FLX_' + dataset + '/'
                file = 'FLX_' + dataset + '_FLUXNET2015_FULLSET_HH'    
                
                try:
                    for file_names in os.listdir(folder_path):
                        if file in file_names:
                            self.data_all = pd.read_csv(folder_path + file_names)
                except FileNotFoundError as fnf_error:
                    print(fnf_error)
            else:
                print('Work in progress')
                return None
        
        
        if dataset == 'syn':
            self.experiment_config['Y'] = 'NEE_syn'
        else:
            self.experiment_config['Y'] = self.experiment_config['Y'] + '_' + self.experiment_config['method'] + '_REF'        
        
        if dataset == 'syn':
                    columns = ['DateTime'] + \
                    self.experiment_config['X'] + \
                    self.experiment_config['W'] + \
                    ['SW_IN'] + \
                    [self.experiment_config['Y']] + \
                    ['RUE_syn', 'GPP_syn', 'Rb_syn', 'RECO_syn', 'GPP_DT', 'GPP_NT', 'RECO_DT', 'RECO_NT']
        else:
            others = ['CO2_F_MDS', 'NEE_CUT_REF', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'RECO_NT_CUT_REF', 'RECO_DT_VUT_REF', 'RECO_DT_CUT_REF', 'GPP_NT_VUT_REF',
                    'GPP_NT_CUT_REF', 'GPP_DT_VUT_REF', 'GPP_DT_CUT_REF']
            others.remove(self.experiment_config['Y'])
            
            columns = ['TIMESTAMP_START'] + \
                    self.experiment_config['X'] + \
                    self.experiment_config['W'] + \
                    [self.experiment_config['T']] + \
                    [self.experiment_config['Y']] + \
                    others
            
            self.data_all['wdefCum'] = wdefcum(self.data_all['LE_F_MDS'], self.data_all['P_F'])

        for parameter in columns:
            if parameter not in self.data_all.columns:
                columns.remove(parameter)
                print(f"{parameter} is not a column of the dataset and hence removed")
                try:
                    self.experiment_config['X'].remove(parameter)
                except:
                    pass
                try:
                    self.experiment_config['W'].remove(parameter)
                except:
                    pass
                try:
                    self.experiment_config['w'].remove(parameter)
                except:
                    pass

        
        data = self.data_all[columns].copy()
        
        for column in columns:
            data = data[data[column] != -9999.000]
                
        
        if dataset != 'syn':
            data['DateTime'] = pd.to_datetime(data['TIMESTAMP_START']+15, format="%Y%m%d%H%M")
        data["Date"] = pd.to_datetime(data['DateTime']).dt.date
        data["Time"] = pd.to_datetime(data['DateTime']).dt.time
        data["Month"] = pd.to_datetime(data['Date']).dt.month
        data["Year"] = pd.to_datetime(data['Date']).dt.year
        data["doy"] = pd.to_datetime(data['DateTime']).dt.dayofyear

        self.data_train = data[data['Year'].isin(range(*self.experiment_config['train_period']))].copy()
        self.data_test = data[data['Year'].isin(range(*self.experiment_config['test_period']))].copy()        
        
        
        
        if self.experiment_config['training_type'] == 'monthly':
            self.X_train_temp = dict()
            self.X_test_temp = dict()
            self.W_train_temp = dict()
            self.W_test_temp = dict()
            self.Y_train_temp = dict()
            self.Y_test_temp = dict()
            self.T_train_temp = dict()
            self.T_test_temp = dict()
            self.est_dict = dict()
            
            for month, month_str in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                month = month + 1
                self.data_train_temp = self.data_train[self.data_train['Month'] == month].copy()
                self.data_test_temp = self.data_test[self.data_test['Month'] == month].copy()
                
                if self.experiment_config['X_normed']:
                    for parameter in self.experiment_config['X']:
                        self.data_train_temp[parameter + '_normed'] = (self.data_train_temp[parameter].loc[:]-self.data_train_temp[parameter].mean())/self.data_train_temp[parameter].std()
                        self.data_test_temp[parameter + '_normed'] = (self.data_test_temp[parameter].loc[:]-self.data_train_temp[parameter].mean())/self.data_train_temp[parameter].std()
                    
                    self.X_train_temp[month] = self.data_train_temp[[parameter + '_normed' for parameter in self.experiment_config['X']]]
                    self.X_test_temp[month] = self.data_test_temp[[parameter + '_normed' for parameter in self.experiment_config['X']]]
                else:
                    self.X_train_temp[month] = self.data_train_temp[self.experiment_config['X']]
                    self.X_test_temp[month] = self.data_test_temp[self.experiment_config['X']]
            
                if self.experiment_config['W_normed']:
                    for parameter in self.experiment_config['W']:
                        self.data_train_temp[parameter + '_normed'] = (self.data_train_temp[parameter].loc[:]-self.data_train_temp[parameter].mean())/self.data_train_temp[parameter].std()
                        self.data_test_temp[parameter + '_normed'] = (self.data_test_temp[parameter].loc[:]-self.data_train_temp[parameter].mean())/self.data_train_temp[parameter].std()
                    self.W_train_temp[month] = self.data_train_temp[[parameter + '_normed' for parameter in self.experiment_config['W']]]
                    self.W_test_temp[month] = self.data_test_temp[[parameter + '_normed' for parameter in self.experiment_config['W']]]
                else:
                    self.W_train_temp[month] = self.data_train_temp[self.experiment_config['W']]
                    self.W_test_temp[month] = self.data_test_temp[self.experiment_config['W']]
                                
                if self.experiment_config['Y_normed']:
                    parameter = self.experiment_config['Y']
                    self.data_train_temp[parameter + '_normed'] = (self.data_train_temp[parameter].loc[:]-self.data_train_temp[parameter].mean())/self.data_train_temp[parameter].std()
                    self.data_test_temp[parameter + '_normed'] = (self.data_test_temp[parameter].loc[:]-self.data_train_temp[parameter].mean())/self.data_train_temp[parameter].std()
                    self.Y_train_temp[month] = self.data_train_temp[parameter + '_normed']
                    self.Y_test_temp[month] = self.data_test_temp[parameter + '_normed']
                else:
                    self.Y_train_temp[month] = self.data_train_temp[self.experiment_config['Y']]
                    self.Y_test_temp[month] = self.data_test_temp[self.experiment_config['Y']]    

                if self.experiment_config['T_normed']:
                    parameter = self.experiment_config['T']
                    self.data_train_temp[parameter + '_normed'] = (self.data_train_temp[parameter].loc[:]-self.data_train_temp[parameter].mean())/self.data_train_temp[parameter].std()
                    self.data_test_temp[parameter + '_normed'] = (self.data_test_temp[parameter].loc[:]-self.data_train_temp[parameter].mean())/self.data_train_temp[parameter].std()   
                    self.T_train_temp[month] = self.data_train_temp[parameter + '_normed']
                    self.T_test_temp[month] = self.data_test_temp[parameter + '_normed']
                else:
                    self.T_train_temp[month] = self.data_train_temp[self.experiment_config['T']]
                    self.T_test_temp[month] = self.data_test_temp[self.experiment_config['T']]     

                if self.experiment_config['delta'] == 'default':
                    delta = 5/self.data_train[self.experiment_config['T']].max()
                else:
                    delta = self.experiment_config['delta']
                
                self.T_train_temp = hyperbolic_transform(self.T_train_temp, delta=delta, transform=self.experiment_config['transform'])
                self.T_test_temp = hyperbolic_transform(self.T_test_temp, delta=delta, transform=self.experiment_config['transform'])


                # Fit the model (quick and dirty)
                if self.experiment_config['model_y'] == 'GradientBoostingRegressor':
                    model_y = GradientBoostingRegressor(n_estimators = self.experiment_config['n_estimators'], 
                                                    min_samples_split = self.experiment_config['min_samples_split'], 
                                                    min_samples_leaf = self.experiment_config['min_samples_leaf'], 
                                                    max_depth = self.experiment_config['max_depth'])
                if self.experiment_config['model_t'] == 'GradientBoostingRegressor':
                    model_t = GradientBoostingRegressor(n_estimators = self.experiment_config['n_estimators'], 
                                                    min_samples_split = self.experiment_config['min_samples_split'], 
                                                    min_samples_leaf = self.experiment_config['min_samples_leaf'], 
                                                    max_depth = self.experiment_config['max_depth'])
                if self.experiment_config['model_final'] == 'GradientBoostingRegressor':
                    model_final = GradientBoostingRegressor(n_estimators = self.experiment_config['n_estimators'], 
                                                    min_samples_split = self.experiment_config['min_samples_split'], 
                                                    min_samples_leaf = self.experiment_config['min_samples_leaf'], 
                                                    max_depth = self.experiment_config['max_depth'])

                self.est_dict[month] = monthly_est()        
                self.est_dict[month].est = NonParamDML(model_y=model_y,
                                model_t=model_t,
                                model_final = model_final, 
                                cv=self.experiment_config['cv'])

                if self.experiment_config['W']:
                    self.est_dict[month].est.fit(self.Y_train_temp[month], self.T_train_temp[month], X=self.X_train_temp[month], W=self.W_train_temp[month], cache_values=True)
                else:
                    self.est_dict[month].est.fit(self.Y_train_temp[month], self.T_train_temp[month], X=self.X_train_temp[month], cache_values=True)

            
                self.est_dict[month].nuisance_scores_y = self.est_dict[month].est.nuisance_scores_y
                self.est_dict[month].nuisance_scores_t = self.est_dict[month].est.nuisance_scores_t
                self.est_dict[month].score_train = self.est_dict[month].est.score_
                if self.experiment_config['W']:
                    self.est_dict[month].score_test = self.est_dict[month].est.score(self.Y_test_temp[month], self.T_test_temp[month], X=self.X_test_temp[month], W=self.W_test_temp[month])
                    self.est_dict[month].nuisance = (self.Y_train_temp[month]-self.est_dict[month].est._cached_values[0][0])-self.est_dict[month].est.const_marginal_effect(self.X_train_temp[month], self.W_train[month])*(self.T_train[month]-self.est_dict[month].est._cached_values[0][1])
                else:
                    self.est_dict[month].score_test = self.est_dict[month].est.score(self.Y_test_temp[month], self.T_test_temp[month], X=self.X_test_temp[month])
                    self.est_dict[month].nuisance = (self.Y_train_temp[month]-self.est_dict[month].est._cached_values[0][0])-self.est_dict[month].est.const_marginal_effect(self.X_train_temp[month])*(self.T_train_temp[month]-self.est_dict[month].est._cached_values[0][1])
            
            
            self.X_train = pd.concat(list(self.X_train_temp.values())).sort_index().values
            self.X_test = pd.concat(list(self.X_test_temp.values())).sort_index().values
            self.Y_train = pd.concat(list(self.Y_train_temp.values())).sort_index().values
            self.Y_test = pd.concat(list(self.Y_test_temp.values())).sort_index().values
            self.T_train = pd.concat(list(self.T_train_temp.values())).sort_index().values
            self.T_test = pd.concat(list(self.T_test_temp.values())).sort_index().values
            if self.experiment_config['W']:
                self.W_train = pd.concat(list(self.W_train_temp.values())).sort_index().values
                self.W_test = pd.concat(list(self.W_test_temp.values())).sort_index().values

        else:
            self.est_dict = dict()
            
            if self.experiment_config['X_normed']:
                for parameter in self.experiment_config['X']:
                    self.data_train[parameter + '_normed'] = (self.data_train[parameter].loc[:]-self.data_train[parameter].mean())/self.data_train[parameter].std()
                    self.data_test[parameter + '_normed'] = (self.data_test[parameter].loc[:]-self.data_train[parameter].mean())/self.data_train[parameter].std()
                self.X_train = self.data_train[[parameter + '_normed' for parameter in self.experiment_config['X']]]
                self.X_test = self.data_test[[parameter + '_normed' for parameter in self.experiment_config['X']]]
            else:
                self.X_train = self.data_train[self.experiment_config['X']]
                self.X_test = self.data_test[self.experiment_config['X']]
            
            if self.experiment_config['W_normed']:
                for parameter in self.experiment_config['W']:
                    self.data_train[parameter + '_normed'] = (self.data_train[parameter].loc[:]-self.data_train[parameter].mean())/self.data_train[parameter].std()
                    self.data_test[parameter + '_normed'] = (self.data_test[parameter].loc[:]-self.data_train[parameter].mean())/self.data_train[parameter].std()
                self.W_train = self.data_train[[parameter + '_normed' for parameter in self.experiment_config['W']]]
                self.W_test = self.data_test[[parameter + '_normed' for parameter in self.experiment_config['W']]]
            else:
                self.W_train = self.data_train[self.experiment_config['W']]
                self.W_test = self.data_test[self.experiment_config['W']]
                            
            if self.experiment_config['Y_normed']:
                parameter = self.experiment_config['Y']
                self.data_train[parameter + '_normed'] = (self.data_train[parameter].loc[:]-self.data_train[parameter].mean())/self.data_train[parameter].std()
                self.data_test[parameter + '_normed'] = (self.data_test[parameter].loc[:]-self.data_train[parameter].mean())/self.data_train[parameter].std()
                self.Y_train = self.data_train[parameter + '_normed']
                self.Y_test = self.data_test[parameter + '_normed']
            else:
                self.Y_train = self.data_train[self.experiment_config['Y']]
                self.Y_test = self.data_test[self.experiment_config['Y']]    

            if self.experiment_config['T_normed']:
                parameter = self.experiment_config['T']
                self.data_train[parameter + '_normed'] = (self.data_train[parameter].loc[:]-self.data_train[parameter].mean())/self.data_train[parameter].std()
                self.data_test[parameter + '_normed'] = (self.data_test[parameter].loc[:]-self.data_train[parameter].mean())/self.data_train[parameter].std()   
                self.T_train = self.data_train[parameter + '_normed']
                self.T_test = self.data_test[parameter + '_normed']
            else:
                self.T_train = self.data_train[self.experiment_config['T']]
                self.T_test = self.data_test[self.experiment_config['T']]     

            if self.experiment_config['w_normed']:
                for parameter in self.experiment_config['w']:
                    self.data_train[parameter + '_normed'] = (self.data_train[parameter].loc[:]-self.data_train[parameter].mean())/self.data_train[parameter].std()
                    self.data_test[parameter + '_normed'] = (self.data_test[parameter].loc[:]-self.data_train[parameter].mean())/self.data_train[parameter].std()

            if self.experiment_config['delta'] == 'default':
                delta = 5/self.T_train.max()
            else:
                delta = self.experiment_config['delta']
            
            self.T_train = hyperbolic_transform(self.T_train, delta=delta, transform=self.experiment_config['transform'])
            self.T_test = hyperbolic_transform(self.T_test, delta=delta, transform=self.experiment_config['transform'])


            # Fit the model (quick and dirty)
            if self.experiment_config['model_y'] == 'GradientBoostingRegressor':
                model_y = GradientBoostingRegressor(n_estimators = self.experiment_config['n_estimators'], 
                                                    min_samples_split = self.experiment_config['min_samples_split'], 
                                                    min_samples_leaf = self.experiment_config['min_samples_leaf'], 
                                                    max_depth = self.experiment_config['max_depth'])
            if self.experiment_config['model_t'] == 'GradientBoostingRegressor':
                model_t = GradientBoostingRegressor(n_estimators = self.experiment_config['n_estimators'], 
                                                    min_samples_split = self.experiment_config['min_samples_split'], 
                                                    min_samples_leaf = self.experiment_config['min_samples_leaf'], 
                                                    max_depth = self.experiment_config['max_depth'])
            if self.experiment_config['model_final'] == 'GradientBoostingRegressor':
                model_final = GradientBoostingRegressor(n_estimators = self.experiment_config['n_estimators'], 
                                                    min_samples_split = self.experiment_config['min_samples_split'], 
                                                    min_samples_leaf = self.experiment_config['min_samples_leaf'], 
                                                    max_depth = self.experiment_config['max_depth'])

            self.est = NonParamDML(model_y=model_y,
                    model_t=model_t,
                    model_final = model_final, 
                    cv=self.experiment_config['cv'])

            if self.experiment_config['W']:
                self.est.fit(self.Y_train, self.T_train, X=self.X_train, W=self.W_train, cache_values=True)
            else:
                self.est.fit(self.Y_train, self.T_train, X=self.X_train, cache_values=True)

            self.nuisance_scores_y = self.est.nuisance_scores_y
            self.nuisance_scores_t = self.est.nuisance_scores_t
            self.score_train = self.est.score_
            if self.experiment_config['W']:
                self.score_test = self.est.score(self.Y_test, self.T_test, X=self.X_test, W=self.W_test)
                self.nuisance = (self.Y_train-self.est._cached_values[0][0])-self.est.const_marginal_effect(self.X_train)*(self.T_train-self.est._cached_values[0][1])
            else:
                self.score_test = self.est.score(self.Y_test, self.T_test, X=self.X_test)
                self.nuisance = (self.Y_train-self.est._cached_values[0][0])-self.est.const_marginal_effect(self.X_train)*(self.T_train-self.est._cached_values[0][1])

        self.gpp, self.reco, self.nee, self.t =  self.get_estimators(training_type=self.experiment_config['training_type'])
        
        
        if self.experiment_config['training_type'] != 'monthly':
            self.data_train['GPP_orth'] = self.gpp(self.X_train, self.T_train)
            #self.data_train['GPP_nuisance'] = self.gpp(self.X_train, self.T_train)
            self.data_test['GPP_orth'] = self.gpp(self.X_test, self.T_test)

            if self.experiment_config['W']:
                #self.data_train['RECO_nuisance'] = self.nuisance
                self.data_train['RECO_orth'] = self.reco(self.X_train, self.T_train, self.W_train)
                self.data_test['RECO_orth'] = self.reco(self.X_test, self.T_test, self.W_test)
                self.data_train['RECO_orth_rest'] = self.Y_train + self.data_train['GPP_orth']
                self.data_test['RECO_orth_rest'] = self.Y_test + self.data_test['GPP_orth'] 

                #self.data_train['NEE_nuisance'] = self.nuisance - self.data_train['GPP_orth']
                self.data_train['NEE_orth'] = self.nee(self.X_train, self.T_train, self.W_train)
                self.data_test['NEE_orth'] = self.nee(self.X_test, self.T_test, self.W_test)

                self.data_train['T_estimate'] = self.t(self.X_train, self.W_train)
                self.data_test['T_estimate'] = self.t(self.X_test, self.W_test)
                
            else:
                #self.data_train['RECO_nuisance'] = self.nuisance
                self.data_train['RECO_orth'] = self.reco(self.X_train, self.T_train)
                self.data_test['RECO_orth'] = self.reco(self.X_test, self.T_test)
                self.data_train['RECO_orth_rest'] = self.Y_train + self.data_train['GPP_orth']
                self.data_test['RECO_orth_rest'] = self.Y_test + self.data_test['GPP_orth'] 

                #self.data_train['NEE_nuisance'] = self.nuisance - self.data_train['GPP_orth']
                self.data_train['NEE_orth'] = self.nee(self.X_train, self.T_train)
                self.data_test['NEE_orth'] = self.nee(self.X_test, self.T_test)

                self.data_train['T_estimate'] = self.t(self.X_train)
                self.data_test['T_estimate'] = self.t(self.X_test)
        else:
            self.data_train['GPP_orth'] = self.gpp(self.X_train, self.T_train, self.data_train['Month'].values)
            #self.data_train['GPP_nuisance'] = self.gpp(self.X_train, self.T_train, self.data_train['Month'])
            self.data_test['GPP_orth'] = self.gpp(self.X_test, self.T_test, self.data_test['Month'].values)

            #self.data_train['RECO_nuisance'] = self.nuisance
            self.data_train['RECO_orth'] = self.reco(self.X_train, self.T_train, self.data_train['Month'].values)
            self.data_test['RECO_orth'] = self.reco(self.X_test, self.T_test, self.data_test['Month'].values)
            self.data_train['RECO_orth_rest'] = self.Y_train + self.data_train['GPP_orth']
            self.data_test['RECO_orth_rest'] = self.Y_test + self.data_test['GPP_orth'] 


            #self.data_train['NEE_nuisance'] = self.nuisance - self.data_train['GPP_orth']
            self.data_train['NEE_orth'] = self.nee(self.X_train, self.T_train, self.data_train['Month'].values)
            self.data_test['NEE_orth'] = self.nee(self.X_test, self.T_test, self.data_test['Month'].values)
            
            self.data_train['T_estimate'] = self.t(self.X_train, self.data_train['Month'].values)
            self.data_test['T_estimate'] = self.t(self.X_test, self.data_test['Month'].values)
        
        
        if self.experiment_config['clean_output']:
            self.data_train['GPP_orth'] = clean_outliers(self.data_train['GPP_orth'])
            #self.data_train['GPP_nuisance'] = self.gpp(self.X_train, self.T_train, self.data_train['Month'])
            self.data_test['GPP_orth'] = clean_outliers(self.data_test['GPP_orth'])

            #self.data_train['RECO_nuisance'] = self.nuisance
            self.data_train['RECO_orth'] = clean_outliers(self.data_train['RECO_orth'])
            self.data_test['RECO_orth'] = clean_outliers(self.data_test['RECO_orth'])
            self.data_train['RECO_orth_rest'] = clean_outliers(self.data_train['RECO_orth_rest'])
            self.data_test['RECO_orth_rest'] = clean_outliers(self.data_test['RECO_orth_rest'])

            #self.data_train['NEE_nuisance'] = self.nuisance - self.data_train['GPP_orth']
            self.data_train['NEE_orth'] = clean_outliers(self.data_train['NEE_orth'])
            self.data_test['NEE_orth'] = clean_outliers(self.data_test['NEE_orth'])
            
            self.data_train['T_estimate'] = clean_outliers(self.data_train['T_estimate'])
            self.data_test['T_estimate'] = clean_outliers(self.data_test['T_estimate'])
            
        if self.experiment_config['w_normed']:
            for parameter in self.experiment_config['w']:
                self.data_train[parameter + '_normed'] = (self.data_train[parameter].loc[:]-self.data_train[parameter].mean())/self.data_train[parameter].std()
                self.data_test[parameter + '_normed'] = (self.data_test[parameter].loc[:]-self.data_train[parameter].mean())/self.data_train[parameter].std()
        
        self.Q10_dict = dict()
        
        if self.experiment_config['w_normed']:
            w = [parameter + '_normed' for parameter in self.experiment_config['w']]
        

        if self.experiment_config['q_10'] == True:

            #nuisance = dict()
            #dfl_nuisance = self.data_train.loc[(self.data_train['RECO_nuisance'] > 0)].copy()
            #dfl_nuisance['RECO_log'] = np.log(dfl_nuisance['RECO_nuisance'])
            #nuisance['inconsistent'] = sum(self.data_train['RECO_nuisance'] <= 0)
            #nuisance['consistent'] = sum(self.data_train['RECO_nuisance'] > 0)

            orth_train = dict()
            dfl_orth_train = self.data_train.loc[(self.data_train['RECO_orth'] > 0)].copy()
            dfl_orth_train['RECO_log'] = np.log(dfl_orth_train['RECO_orth'])
            orth_train['inconsistent'] = sum(self.data_train['RECO_orth'] <= 0)
            orth_train['consistent'] = sum(self.data_train['RECO_orth'] > 0)
            
            orth_rest_train = dict()
            dfl_orth_rest_train = self.data_train.loc[(self.data_train['RECO_orth_rest'] > 0)].copy()
            dfl_orth_rest_train['RECO_rest_log'] = np.log(dfl_orth_rest_train['RECO_orth_rest'])
            orth_rest_train['inconsistent'] = sum(self.data_train['RECO_orth_rest'] <= 0)
            orth_rest_train['consistent'] = sum(self.data_train['RECO_orth_rest'] > 0)

            orth_test = dict()
            dfl_orth_test = self.data_test.loc[(self.data_test['RECO_orth'] > 0)].copy()
            dfl_orth_test['RECO_log'] = np.log(dfl_orth_test['RECO_orth'])
            orth_test['inconsistent'] = sum(self.data_test['RECO_orth'] <= 0)
            orth_test['consistent'] = sum(self.data_test['RECO_orth'] > 0)

            # Fit the model (quick and dirty)
            # if self.experiment_config['model_m'] == 'GradientBoostingRegressor':
            #    model_m = GradientBoostingRegressor()
            # if self.experiment_config['model_g'] == 'GradientBoostingRegressor':
            #    model_g = GradientBoostingRegressor()

            #dfl_nuisance['TA_scaled'] =  (dfl_nuisance['TA_F']-15)/10

            #nuisance['obj_dml_data'] = dml.DoubleMLData(dfl_nuisance,  y_col='RECO_log',
            #                        d_cols='TA_scaled',
            #                        x_cols=w,
            #                        use_other_treat_as_covariate=False)

            #nuisance['dml_plr_obj'] = dml.DoubleMLPLR(nuisance['obj_dml_data'], model_g, model_m, dml_procedure=self.experiment_config['dml_procedure'], n_folds=self.experiment_config['n_folds'])
            
            #nuisance['dml_plr_obj'].fit()
            #nuisance['q10_nuisance'] = np.exp(nuisance['dml_plr_obj'].coef)
            
            #self.Q10_dict['nuisance'] = nuisance
            
            # Fit the model (quick and dirty)
            if self.experiment_config['model_m'] == 'GradientBoostingRegressor':
                model_m = GradientBoostingRegressor(n_estimators = self.experiment_config['n_estimators'], 
                                                    min_samples_split = self.experiment_config['min_samples_split'], 
                                                    min_samples_leaf = self.experiment_config['min_samples_leaf'], 
                                                    max_depth = self.experiment_config['max_depth'])
            if self.experiment_config['model_g'] == 'GradientBoostingRegressor':
                model_g = GradientBoostingRegressor(n_estimators = self.experiment_config['n_estimators'], 
                                                    min_samples_split = self.experiment_config['min_samples_split'], 
                                                    min_samples_leaf = self.experiment_config['min_samples_leaf'], 
                                                    max_depth = self.experiment_config['max_depth'])


            if dataset == 'syn':
                dfl_orth_train['TA_scaled'] =  (dfl_orth_train['TA']-15)/10
            else:
                dfl_orth_train['TA_scaled'] =  (dfl_orth_train['TA_F']-15)/10

            orth_train['obj_dml_data'] = dml.DoubleMLData(dfl_orth_train,  y_col='RECO_log',
                                    d_cols='TA_scaled',
                                    x_cols=w,
                                    use_other_treat_as_covariate=False)

            orth_train['dml_plr_obj'] = dml.DoubleMLPLR(orth_train['obj_dml_data'], model_g, model_m, dml_procedure=self.experiment_config['dml_procedure'], n_folds=self.experiment_config['n_folds'])
            
            orth_train['dml_plr_obj'].fit()
            orth_train['q10'] = np.exp(orth_train['dml_plr_obj'].coef)
            
            self.Q10_dict['orth_train'] = orth_train

            if self.experiment_config['model_m'] == 'GradientBoostingRegressor':
                model_m = GradientBoostingRegressor(n_estimators = self.experiment_config['n_estimators'], 
                                                    min_samples_split = self.experiment_config['min_samples_split'], 
                                                    min_samples_leaf = self.experiment_config['min_samples_leaf'], 
                                                    max_depth = self.experiment_config['max_depth'])
            if self.experiment_config['model_g'] == 'GradientBoostingRegressor':
                model_g = GradientBoostingRegressor(n_estimators = self.experiment_config['n_estimators'], 
                                                    min_samples_split = self.experiment_config['min_samples_split'], 
                                                    min_samples_leaf = self.experiment_config['min_samples_leaf'], 
                                                    max_depth = self.experiment_config['max_depth'])


            if dataset == 'syn':
                dfl_orth_rest_train['TA_scaled'] =  (dfl_orth_rest_train['TA']-15)/10
            else:
                dfl_orth_rest_train['TA_scaled'] =  (dfl_orth_rest_train['TA_F']-15)/10

            orth_rest_train['obj_dml_data'] = dml.DoubleMLData(dfl_orth_rest_train,  y_col='RECO_rest_log',
                                    d_cols='TA_scaled',
                                    x_cols=w,
                                    use_other_treat_as_covariate=False)

            orth_rest_train['dml_plr_obj'] = dml.DoubleMLPLR(orth_rest_train['obj_dml_data'], model_g, model_m, dml_procedure=self.experiment_config['dml_procedure'], n_folds=self.experiment_config['n_folds'])
            
            orth_rest_train['dml_plr_obj'].fit()
            orth_rest_train['q10'] = np.exp(orth_rest_train['dml_plr_obj'].coef)
            
            self.Q10_dict['orth_rest_train'] = orth_rest_train

            
            # Fit the model (quick and dirty)
            if self.experiment_config['model_m'] == 'GradientBoostingRegressor':
                model_m = GradientBoostingRegressor(n_estimators = self.experiment_config['n_estimators'], 
                                                    min_samples_split = self.experiment_config['min_samples_split'], 
                                                    min_samples_leaf = self.experiment_config['min_samples_leaf'], 
                                                    max_depth = self.experiment_config['max_depth'])
            if self.experiment_config['model_g'] == 'GradientBoostingRegressor':
                model_g = GradientBoostingRegressor(n_estimators = self.experiment_config['n_estimators'], 
                                                    min_samples_split = self.experiment_config['min_samples_split'], 
                                                    min_samples_leaf = self.experiment_config['min_samples_leaf'], 
                                                    max_depth = self.experiment_config['max_depth'])

            if dataset == 'syn':
                dfl_orth_test['TA_scaled'] =  (dfl_orth_test['TA']-15)/10
            else:
                dfl_orth_test['TA_scaled'] =  (dfl_orth_test['TA_F']-15)/10

            orth_test['obj_dml_data'] = dml.DoubleMLData(dfl_orth_test,  y_col='RECO_log',
                                    d_cols='TA_scaled',
                                    x_cols=w,
                                    use_other_treat_as_covariate=False)

            orth_test['dml_plr_obj'] = dml.DoubleMLPLR(orth_test['obj_dml_data'], model_g, model_m, dml_procedure=self.experiment_config['dml_procedure'], n_folds=self.experiment_config['n_folds'])
            
            orth_test['dml_plr_obj'].fit()
            orth_test['q10'] = np.exp(orth_test['dml_plr_obj'].coef)
            
            self.Q10_dict['orth_test'] = orth_test
            
        original_stdout = sys.stdout
        
        for data_type, data_set in zip(['train', 'test'], [self.data_train, self.data_test]):
            #try:
            os.mkdir(self.PATH + '/results/' + data_type)
            #except:
            #    print(f'{data_type} folder already exists!')
            
                
            with open(self.PATH + '/results/' + data_type +'/results.txt', 'w') as f:
                sys.stdout = f

                print(f'Results for dataset {data_type}\n')                
                if data_type == 'train':
                    if self.experiment_config['training_type'] != 'monthly':
                        print('Fitting results:\n')
                        print(f'SW_IN nuisance score (R^2): {self.est.nuisance_scores_t}\n') 
                        print(f'NEE nuisance score (R^2): {self.est.nuisance_scores_y}\n')
                        print(f'fitting of the residuals for final model (MSE): {self.est.score_}\n')
                        print('\n')
                    else:
                        for month, month_str in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                            month = month + 1
                            print(f'Fitting results for {month_str}:\n')
                            print(f'SW_IN nuisance score (R^2): {self.est_dict[month].est.nuisance_scores_t}\n') 
                            print(f'NEE nuisance score (R^2): {self.est_dict[month].est.nuisance_scores_y}\n')
                            print(f'fitting of the residuals for final model (MSE): {self.est_dict[month].est.score_}\n')
                            print('\n')

                if dataset == 'syn':
                    ref = ''
                else:
                    ref = '_' + self.experiment_config['method'] + '_REF'

                if dataset != 'syn':
                    print('R2 to the other estimators:\n')
                    print('GPP:\n')
                    print(f'DT vs orth: {r2_score(data_set["GPP_DT"+ ref], data_set["GPP_orth"]):.4f}') 
                    print(f'NT vs orth: {r2_score(data_set["GPP_NT"+ ref], data_set["GPP_orth"]):.4f}') 
                    print(f'DT vs NT: {r2_score(data_set["GPP_DT"+ ref], data_set["GPP_NT"+ ref]):.4f}\n') 

                    print('RECO pred:\n')
                    #if data_type == 'train':
                    #    print('Nuisance estimator:')
                    #    print(f'DT vs nuisance: {r2_score(data_set["RECO_DT"+ ref], data_set["RECO_nuisance"]):.4f}') 
                    #    print(f'NT vs nuisance: {r2_score(data_set["RECO_NT"+ ref], data_set["RECO_nuisance"]):.4f}') 
                    #    print(f'DT vs NT: {r2_score(data_set["RECO_DT" + ref], data_set["RECO_NT"+ ref]):.4f}\n') 
                    
                    print('RECO predictor:')
                    print(f'DT vs orth: {r2_score(data_set["RECO_DT"+ ref], data_set["RECO_orth"]):.4f}') 
                    print(f'NT vs orth: {r2_score(data_set["RECO_NT"+ ref], data_set["RECO_orth"]):.4f}') 
                    print(f'DT vs NT: {r2_score(data_set["RECO_DT"+ ref], data_set["RECO_NT"+ ref]):.4f}\n') 
                                                                
                
                    print('RECO rest:\n')
                    #if data_type == 'train':
                    #    print('Nuisance estimator:')
                    #    print(f'DT vs nuisance: {r2_score(data_set["RECO_DT"+ ref], data_set["RECO_nuisance"]):.4f}') 
                    #    print(f'NT vs nuisance: {r2_score(data_set["RECO_NT"+ ref], data_set["RECO_nuisance"]):.4f}') 
                    #    print(f'DT vs NT: {r2_score(data_set["RECO_DT" + ref], data_set["RECO_NT"+ ref]):.4f}\n') 
                    
                    print('RECO predictor:')
                    print(f'DT vs orth: {r2_score(data_set["RECO_DT"+ ref], data_set["RECO_orth_rest"]):.4f}') 
                    print(f'NT vs orth: {r2_score(data_set["RECO_NT"+ ref], data_set["RECO_orth_rest"]):.4f}') 
                    print(f'DT vs NT: {r2_score(data_set["RECO_DT"+ ref], data_set["RECO_NT"+ ref]):.4f}\n') 
                        
                    print('NEE:\n')
                    #if data_type == 'train':
                    #    print('Nuisance estimator:')
                    #    print(f'NEE vs nuisance: {r2_score(data_set["NEE" + ref], data_set["NEE_nuisance"]):.4f}\n')                     
                            
                    data_set['NEE_DT' + ref] = data_set["RECO_DT" + ref] - data_set["GPP_DT" + ref]
                    data_set['NEE_NT' + ref] = data_set["RECO_NT" + ref] - data_set["GPP_NT" + ref]
                    
                    print('NEE predictor:')
                    print(f'NEE vs orth: {r2_score(data_set["NEE"+ ref], data_set["NEE_orth"]):.4f}') 
                    print(f'NEE vs DT: {r2_score(data_set["NEE" + ref], data_set["NEE_DT" + ref]):.4f}') 
                    print(f'NEE vs NT: {r2_score(data_set["NEE" + ref], data_set["NEE_NT" + ref]):.4f}\n') 

                    if data_type == 'train' and self.experiment_config['q_10']:             
                        print('Q10 results')
                        print(self.Q10_dict)
                    
                    sys.stdout = original_stdout
                else:
                    print('R2 to the other estimators:\n')
                    print('GPP:\n')
                    print(f'DT vs orth: {r2_score(data_set["GPP_DT"+ ref], data_set["GPP_orth"]):.4f}') 
                    print(f'NT vs orth: {r2_score(data_set["GPP_NT"+ ref], data_set["GPP_orth"]):.4f}') 
                    print(f'DT vs NT: {r2_score(data_set["GPP_DT"+ ref], data_set["GPP_NT"+ ref]):.4f}') 
                    print(f'GT vs DT: {r2_score(data_set["GPP_syn"+ ref], data_set["GPP_DT"+ ref]):.4f}') 
                    print(f'GT vs NT: {r2_score(data_set["GPP_syn"+ ref], data_set["GPP_NT"+ ref]):.4f}') 
                    print(f'GT vs orth: {r2_score(data_set["GPP_syn"+ ref], data_set["GPP_orth"]):.4f}\n') 


                    print('RECO:\n')
                    #if data_type == 'train':
                    #    print('Nuisance estimator:')
                    #    print(f'DT vs nuisance: {r2_score(data_set["RECO_DT"+ ref], data_set["RECO_nuisance"]):.4f}') 
                    #    print(f'NT vs nuisance: {r2_score(data_set["RECO_NT"+ ref], data_set["RECO_nuisance"]):.4f}') 
                    #    print(f'DT vs NT: {r2_score(data_set["RECO_DT" + ref], data_set["RECO_NT"+ ref]):.4f}\n') 
                    
                    print('RECO predictor:')
                    print(f'DT vs orth: {r2_score(data_set["RECO_DT"+ ref], data_set["RECO_orth"]):.4f}') 
                    print(f'NT vs orth: {r2_score(data_set["RECO_NT"+ ref], data_set["RECO_orth"]):.4f}') 
                    print(f'DT vs NT: {r2_score(data_set["RECO_DT"+ ref], data_set["RECO_NT"+ ref]):.4f}\n') 
                    print(f'GT vs DT: {r2_score(data_set["RECO_syn"+ ref], data_set["RECO_DT"+ ref]):.4f}') 
                    print(f'GT vs NT: {r2_score(data_set["RECO_syn"+ ref], data_set["RECO_NT"+ ref]):.4f}') 
                    print(f'GT vs orth: {r2_score(data_set["RECO_syn"+ ref], data_set["RECO_orth"]):.4f}\n') 
                    print(f'GT vs orth_rest: {r2_score(data_set["RECO_syn"+ ref], data_set["RECO_orth_rest"]):.4f}\n') 

                
                    print('NEE:\n')
                    #if data_type == 'train':
                    #    print('Nuisance estimator:')
                    #    print(f'NEE vs nuisance: {r2_score(data_set["NEE" + ref], data_set["NEE_nuisance"]):.4f}\n')                     
                            
                    data_set['NEE_DT' + ref] = data_set["RECO_DT" + ref] - data_set["GPP_DT" + ref]
                    data_set['NEE_NT' + ref] = data_set["RECO_NT" + ref] - data_set["GPP_NT" + ref]
                    
                    print('NEE predictor:')
                    print(f'NEE vs orth: {r2_score(data_set["NEE_syn"], data_set["NEE_orth"]):.4f}') 
                    print(f'NEE vs DT: {r2_score(data_set["NEE_syn"], data_set["NEE_DT" + ref]):.4f}') 
                    print(f'NEE vs NT: {r2_score(data_set["NEE_syn"], data_set["NEE_NT" + ref]):.4f}\n') 

                    if data_type == 'train' and self.experiment_config['q_10']:             
                        print('Q10 results')
                        print(self.Q10_dict)
                    
                    sys.stdout = original_stdout
            
            
        self.data_train = self.data_train.reset_index()    
        self.data_test = self.data_test.reset_index()    

        ## Nuisance estimators of fitting
        fig, ax = plt.subplots(1, 2, figsize = (20, 10))

        x, y = self.data_train[self.experiment_config['T']], self.data_train['T_estimate']
        density_scatter(x,y,bins = [30,30], ax=ax[0], fig=fig, cmap='viridis')
        ax[0].plot([x.min(), x.max()], [y.min(), y.max()], color = 'red')
        ax[0].set_xlim(x.min(), x.max())
        ax[0].set_ylim(y.min(), y.max())
        ax[0].set_ylabel('Estimated SW_IN')
        ax[0].set_xlabel('True SW_IN')
        ax[0].set_title('True SW_IN vs averaged estimators')

        x, y = self.data_train[self.experiment_config['Y']], self.data_train['NEE_orth']
        density_scatter(x, y, bins = [30,30], ax=ax[1], fig=fig, cmap='viridis')
        ax[1].set_xlim(x.min(), x.max())
        ax[1].set_ylim(y.min(), y.max())
        ax[1].plot([x.min(),x.max()], [y.min(),y.max()], color = 'red')
        ax[1].set_title('True NEE vs averaged estimators')
        ax[1].set_ylabel('Estimated NEE')
        ax[1].set_xlabel('True NEE')

        fig.savefig(self.PATH + "/results/train/nuisance_estimation.png", bbox_inches='tight')    
        
        if dataset != 'syn':      
            ref = '_' + self.experiment_config['method'] + '_REF'
        else:
            ref = ''
        months = ["Jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sep", "Oct", "Nov", "Dec"]


        if dataset != 'syn':
            for flux, [data_type, data_set, time_window] in itertools.product(['GPP', 'RECO', 'NEE'], zip(['train', 'test'], 
                        [self.data_train, self.data_test], [self.experiment_config['train_period'], self.experiment_config['test_period']])):

                data_set['NEE_DT' + ref] = data_set["RECO_DT" + ref] - data_set["GPP_DT" + ref]
                data_set['NEE_NT' + ref] = data_set["RECO_NT" + ref] - data_set["GPP_NT" + ref]

                for year in range(*time_window):
                    fig, axes = plt.subplots(3,4, figsize=(10,10), sharex=True, sharey=True)
                    
                    for month, ax in enumerate(axes.flatten()):
                        df_temp = data_set[(data_set["Month"] == month+1) & (data_set["Year"] == year)]
                        DT = df_temp.groupby("Time")[flux + "_DT_" + self.experiment_config['method'] + "_REF"].mean()
                        NT = df_temp.groupby("Time")[flux + "_NT_" + self.experiment_config['method'] + "_REF"].mean() 
                        orth = df_temp.groupby("Time")[flux + "_orth"].mean()
                        if flux == 'RECO':
                            orth_rest = df_temp.groupby("Time")[flux + "_orth_rest"].mean()
                        
                        
                        #if data_type == "train" and flux != "GPP":
                        #    nuisance = df_temp.groupby("Time")[flux + "_nuisance"].mean()
                        if flux == "NEE":
                            GT = df_temp.groupby("Time")[flux + "_" + self.experiment_config['method'] + "_REF"].mean() 
                        
                        ax.plot(DT.values, color = "orange", label = "Daytime method")
                        ax.plot(NT.values, color = "black", label = "Nighttime method")
                        ax.plot(orth.values, color = "green", label = "Orthogonal ML")
                        #if flux == 'RECO':
                        #    ax.plot(orth_rest.values, color = "blue", label = "Orthogonal ML rest")

                        #if data_type == "train" and flux != "GPP":
                        #    ax.plot(nuisance.values, color = "green", label = "Orthogonal from nuisances")
                        
                        if flux == "NEE":
                            ax.plot(GT.values, color = "yellow", label = "NEE ground truth")
                            
                        handles, labels = ax.get_legend_handles_labels()

                        fig.legend(handles, labels, loc=(0.45, 0.8), fontsize=15)

                        ax.yaxis.set_tick_params(labelsize=15)
                        
                        ax.set_xticks(list(range(0,48,8)))
                        ax.set_xticklabels(list(range(0,24,4)), fontsize = 15)
                        if month // 4 == 2:
                            ax.set_xlabel("Day time", fontsize = 15)
                        if month % 4 == 0:
                            ax.set_ylabel(flux, fontsize = 15)
                        ax.set_title(months[month], fontsize=15)
                        
                        if flux == 'NEE':
                            ax.set_ylim([-20, 20])
                        elif flux == 'GPP':
                            ax.set_ylim([-2, 20])
                        elif flux == 'RECO':
                            ax.set_ylim([-2, 8])

                    fig.suptitle(f'Comparison of the predicted {flux} for different flux partitioning methods in {self.experiment_config["site"]} in {year}', fontsize =15)
                    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
                    fig.savefig(f'{self.PATH}/results/{data_type}/{flux}{year}.pdf', bbox_inches='tight')         
            
            
            for flux, [data_type, version, data_set] in itertools.product(['GPP', 'RECO', 'NEE'], zip(['train', 'test'], ['_orth', '_orth'],
                        [self.data_train, self.data_train, self.data_test])):

                #if version == '_nuisance' and data_type == 'test':
                #    continue
                
                if flux == 'RECO':
                    version = '_orth_rest'
                    fig, ax = plt.subplots(1,3, figsize=(20, 9))

                    figure1 = ax[0].scatter(data_set[flux + version], y = data_set[flux + '_DT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                    ax[0].plot([-25,25],[-25,25], linestyle = "--", color="black")
                    divider = make_axes_locatable(ax[0])
                    cax = divider.append_axes('bottom', size='5%', pad=0.7)
                    fig.colorbar(figure1, cax=cax, orientation='horizontal', label="doy")

                    linreg  = LinearRegression()
                    linreg.fit(data_set[flux + version].values.reshape(-1,1), y = data_set[flux + '_DT'+ ref].values.reshape(-1,1))
                    ax[0].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color="blue")

                    ax[0].text(data_set[flux + version].min(),data_set[flux + '_DT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_DT'+ ref], data_set[flux + version]):.2f}$")

                    ax[0].set_xlim([data_set[flux + version].min()-1, data_set[flux + version].max()+1])
                    ax[0].set_ylim([data_set[flux + '_DT'+ ref].min()-1, data_set[flux + '_DT'+ ref].max()+1])
                    ax[0].set_xlabel(flux + version)
                    ax[0].set_ylabel(flux + '_DT_' + ref)           
                            

                    figure2 = ax[1].scatter(data_set[flux + version], y = data_set[flux + '_NT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                    ax[1].plot([-25,25],[-25,25], linestyle = "--", color="black")
                    divider = make_axes_locatable(ax[1])
                    cax = divider.append_axes('bottom', size='5%', pad=0.7)
                    cbar = fig.colorbar(figure2, cax=cax, orientation='horizontal',label="doy")


                    linreg  = LinearRegression()
                    linreg.fit(data_set[flux + version].values.reshape(-1,1), y = data_set[flux + '_NT'+ ref].values.reshape(-1,1))
                    ax[1].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color="blue")

                    ax[1].text(data_set[flux + version].min(),data_set[flux + '_NT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_NT'+ ref], data_set[flux + version]):.2f}$")
                            
                    ax[1].set_xlim([data_set[flux + version].min()-1, data_set[flux + version].max()+1])
                    ax[1].set_ylim([data_set[flux + '_NT'+ ref].min()-1, data_set[flux + '_NT'+ ref].max()+1])
                    ax[1].set_xlabel(flux + version)
                    ax[1].set_ylabel(flux + '_NT'+ ref)

                            

                    figure3 = ax[2].scatter(data_set[flux + '_NT'+ ref], y = data_set[flux + '_DT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                    ax[2].plot([-25,25],[-25,25], linestyle = "--", color="black")
                    divider = make_axes_locatable(ax[2])
                    cax = divider.append_axes('bottom', size='5%', pad=0.7)
                    fig.colorbar(figure3, cax=cax, orientation='horizontal', label="doy")

                    linreg  = LinearRegression()
                    linreg.fit(data_set[flux + '_NT'+ ref].values.reshape(-1,1), y = data_set[flux + '_DT'+ ref].values.reshape(-1,1))
                    ax[2].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color = "blue")

                    ax[2].text(data_set[flux + '_NT'+ ref].min(),data_set[flux + '_DT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_DT'+ ref], data_set[flux + '_NT'+ ref]):.2f}$")           
                            
                    ax[2].set_xlim([data_set[flux + '_NT'+ ref].min()-1, data_set[flux + '_NT'+ ref].max()+1])
                    ax[2].set_ylim([data_set[flux + '_DT'+ ref].min()-1, data_set[flux + '_DT'+ ref].max()+1])
                    ax[2].set_xlabel(flux + '_NT'+ ref)
                    ax[2].set_ylabel(flux + '_DT'+ ref)
                            
                    fig.suptitle(f"Comparison of different estimations of flux {flux} in {self.experiment_config['site']}")
                    fig.tight_layout()
                    fig.savefig(f'{self.PATH}/results/{data_type}/{flux}_{version}_scatter.png', bbox_inches='tight')                 

                version = '_orth'
                fig, ax = plt.subplots(1,3, figsize=(20, 9))

                figure1 = ax[0].scatter(data_set[flux + version], y = data_set[flux + '_DT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                ax[0].plot([-25,25],[-25,25], linestyle = "--", color="black")
                divider = make_axes_locatable(ax[0])
                cax = divider.append_axes('bottom', size='5%', pad=0.7)
                fig.colorbar(figure1, cax=cax, orientation='horizontal', label="doy")

                linreg  = LinearRegression()
                linreg.fit(data_set[flux + version].values.reshape(-1,1), y = data_set[flux + '_DT'+ ref].values.reshape(-1,1))
                ax[0].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color="blue")

                ax[0].text(data_set[flux + version].min(),data_set[flux + '_DT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_DT'+ ref], data_set[flux + version]):.2f}$")

                ax[0].set_xlim([data_set[flux + version].min()-1, data_set[flux + version].max()+1])
                ax[0].set_ylim([data_set[flux + '_DT'+ ref].min()-1, data_set[flux + '_DT'+ ref].max()+1])
                ax[0].set_xlabel(flux + version)
                ax[0].set_ylabel(flux + '_DT_' + ref)           
                        

                figure2 = ax[1].scatter(data_set[flux + version], y = data_set[flux + '_NT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                ax[1].plot([-25,25],[-25,25], linestyle = "--", color="black")
                divider = make_axes_locatable(ax[1])
                cax = divider.append_axes('bottom', size='5%', pad=0.7)
                cbar = fig.colorbar(figure2, cax=cax, orientation='horizontal',label="doy")


                linreg  = LinearRegression()
                linreg.fit(data_set[flux + version].values.reshape(-1,1), y = data_set[flux + '_NT'+ ref].values.reshape(-1,1))
                ax[1].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color="blue")

                ax[1].text(data_set[flux + version].min(),data_set[flux + '_NT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_NT'+ ref], data_set[flux + version]):.2f}$")
                        
                ax[1].set_xlim([data_set[flux + version].min()-1, data_set[flux + version].max()+1])
                ax[1].set_ylim([data_set[flux + '_NT'+ ref].min()-1, data_set[flux + '_NT'+ ref].max()+1])
                ax[1].set_xlabel(flux + version)
                ax[1].set_ylabel(flux + '_NT'+ ref)

                        

                figure3 = ax[2].scatter(data_set[flux + '_NT'+ ref], y = data_set[flux + '_DT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                ax[2].plot([-25,25],[-25,25], linestyle = "--", color="black")
                divider = make_axes_locatable(ax[2])
                cax = divider.append_axes('bottom', size='5%', pad=0.7)
                fig.colorbar(figure3, cax=cax, orientation='horizontal', label="doy")

                linreg  = LinearRegression()
                linreg.fit(data_set[flux + '_NT'+ ref].values.reshape(-1,1), y = data_set[flux + '_DT'+ ref].values.reshape(-1,1))
                ax[2].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color = "blue")

                ax[2].text(data_set[flux + '_NT'+ ref].min(),data_set[flux + '_DT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_DT'+ ref], data_set[flux + '_NT'+ ref]):.2f}$")           
                        
                ax[2].set_xlim([data_set[flux + '_NT'+ ref].min()-1, data_set[flux + '_NT'+ ref].max()+1])
                ax[2].set_ylim([data_set[flux + '_DT'+ ref].min()-1, data_set[flux + '_DT'+ ref].max()+1])
                ax[2].set_xlabel(flux + '_NT'+ ref)
                ax[2].set_ylabel(flux + '_DT'+ ref)
                        
                fig.suptitle(f"Comparison of different estimations of flux {flux} in {self.experiment_config['site']}")
                fig.tight_layout()
                fig.savefig(f'{self.PATH}/results/{data_type}/{flux}_{version}_scatter.png', bbox_inches='tight')   
                    
        
            for flux, [data_type, data_set, time_window] in itertools.product(['GPP', 'RECO', 'NEE'], zip(['train', 'test'], 
                        [self.data_train, self.data_test], [self.experiment_config['train_period'], self.experiment_config['test_period']])):
                
                for year in range(*time_window):
                    fig, axes = plt.subplots(12,1, figsize=(30,40), sharex=True, sharey=True)

                    for month, ax in enumerate(axes.flatten()):
                        df_temp = data_set[(data_set["Month"] == month+1) & (data_set["Year"] == year)]
                        DT = df_temp[flux + "_DT"+ ref]
                        NT = df_temp[flux +"_NT"+ ref]
                        orth = df_temp[flux +"_orth"]
                        if flux == 'NEE':
                            GT = df_temp[flux + ref]
                                                    
                        if flux == 'RECO':
                            orth_rest = df_temp[flux +"_orth_rest"]
                        
                        #if data_type == 'train':
                        #    nuisance = df_temp[flux +"_nuisance"]
                        
                        ax.plot(DT.values, color = "orange", label = "Daytime method")
                        ax.plot(NT.values, color = "black", label = "Nighttime method")
                        ax.plot(orth.values, color = "green", label = "Orthogonal ML")
                        
                        if flux == 'NEE':
                            ax.plot(GT.values, color = "yellow", label = "synthetic gronud truth")
                        if flux == 'RECO':
                            ax.plot(orth_rest.values, color = "blue", label = "Orthogonal ML rest")
                        
                        #if data_type == 'train':
                        #    ax.plot(nuisance.values, color = "green", label = "Orthogonal nuisance")
                        if flux == 'NEE':
                            ax.set_ylim([-20, 20])
                        elif flux == 'GPP':
                            ax.set_ylim([-2, 20])
                        elif flux == 'RECO':
                            ax.set_ylim([-2, 8])


                        handles, labels = ax.get_legend_handles_labels()
                        fig.legend(handles, labels, loc=(0.45, 0.87))

                    fig.suptitle(f'Comparison of the predicted {flux} in monthly curves for different flux partitioning methods in {self.experiment_config["site"]} in {year}')
                    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
                    fig.savefig(f'{self.PATH}/results/{data_type}/monthly{flux}{year}.pdf', bbox_inches='tight')
        else:
            for flux, [data_type, data_set, time_window] in itertools.product(['GPP', 'RECO', 'NEE'], zip(['train', 'test'], 
                        [self.data_train, self.data_test], [self.experiment_config['train_period'], self.experiment_config['test_period']])):

                data_set['NEE_DT' + ref] = data_set["RECO_DT" + ref] - data_set["GPP_DT" + ref]
                data_set['NEE_NT' + ref] = data_set["RECO_NT" + ref] - data_set["GPP_NT" + ref]

                for year in range(*time_window):
                    fig, axes = plt.subplots(3,4, figsize=(10,10), sharex=True, sharey=True)
                    
                    for month, ax in enumerate(axes.flatten()):
                        df_temp = data_set[(data_set["Month"] == month+1) & (data_set["Year"] == year)]
                        DT = df_temp.groupby("Time")[flux + "_DT"].mean()
                        NT = df_temp.groupby("Time")[flux + "_NT"].mean() 
                        orth = df_temp.groupby("Time")[flux + "_orth"].mean()
                        GT = df_temp.groupby("Time")[flux + "_syn"].mean()
                        if flux == 'RECO':
                            orth_rest = df_temp.groupby("Time")[flux + "_orth_rest"].mean()
                        
                        #if data_type == "train" and flux != "GPP":
                        #    nuisance = df_temp.groupby("Time")[flux + "_nuisance"].mean()
                        
                        ax.plot(DT.values, color = "orange", label = "Daytime method")
                        ax.plot(NT.values, color = "black", label = "Nighttime method")
                        ax.plot(orth.values, color = "green", label = "Orthogonal ML")
                        if flux == 'RECO':
                            ax.plot(orth_rest.values, color = "blue", label = "Orthogonal ML rest")
                        #if data_type == "train" and flux != "GPP":
                        #    ax.plot(nuisance.values, color = "green", label = "Orthogonal from nuisances")
                        
                        ax.plot(GT.values, color = "yellow", label = "synthetic ground truth")
                            
                        handles, labels = ax.get_legend_handles_labels()

                        fig.legend(handles, labels, loc=(0.45, 0.87))

                        ax.set_xticks(list(range(0,48,8)))
                        ax.set_xticklabels(list(range(0,24,4)), fontsize = 10)
                        if month // 4 == 2:
                            ax.set_xlabel("Day time")
                        if month % 4 == 0:
                            ax.set_ylabel(flux)
                        ax.set_title(months[month])
                        
                        if flux == 'NEE':
                            ax.set_ylim([-20, 20])
                        elif flux == 'GPP':
                            ax.set_ylim([-2, 20])
                        elif flux == 'RECO':
                            ax.set_ylim([-2, 8])

                    fig.suptitle(f'Comparison of the predicted {flux} for different flux partitioning methods in {self.experiment_config["site"]} in {year}')
                    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
                    fig.savefig(f'{self.PATH}/results/{data_type}/{flux}{year}.pdf', bbox_inches='tight')         
            
            
            for flux, [data_type, version, data_set] in itertools.product(['GPP', 'RECO', 'NEE'], zip(['train', 'test'], ['_orth', '_orth'],
                        [self.data_train, self.data_train, self.data_test])):

                #if version == '_nuisance' and data_type == 'test':
                #    continue
                if flux == 'RECO':
                    version = '_orth_rest'
                    fig, ax = plt.subplots(2,3, figsize=(20, 9))

                    figure1 = ax[0,0].scatter(data_set[flux + version], y = data_set[flux + '_DT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                    ax[0,0].plot([-25,25],[-25,25], linestyle = "--", color="black")
                    divider = make_axes_locatable(ax[0,0])
                    cax = divider.append_axes('bottom', size='5%', pad=0.7)
                    fig.colorbar(figure1, cax=cax, orientation='horizontal', label="doy")

                    linreg  = LinearRegression()
                    linreg.fit(data_set[flux + version].values.reshape(-1,1), y = data_set[flux + '_DT'+ ref].values.reshape(-1,1))
                    ax[0,0].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color="blue")

                    ax[0,0].text(data_set[flux + version].min(),data_set[flux + '_DT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_DT'+ ref], data_set[flux + version]):.2f}$")

                    ax[0,0].set_xlim([data_set[flux + version].min()-1, data_set[flux + version].max()+1])
                    ax[0,0].set_ylim([data_set[flux + '_DT'+ ref].min()-1, data_set[flux + '_DT'+ ref].max()+1])
                    ax[0,0].set_xlabel(flux + version)
                    ax[0,0].set_ylabel(flux + '_DT_' + ref)           
                            

                    figure2 = ax[0,1].scatter(data_set[flux + version], y = data_set[flux + '_NT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                    ax[0,1].plot([-25,25],[-25,25], linestyle = "--", color="black")
                    divider = make_axes_locatable(ax[0,1])
                    cax = divider.append_axes('bottom', size='5%', pad=0.7)
                    cbar = fig.colorbar(figure2, cax=cax, orientation='horizontal',label="doy")


                    linreg  = LinearRegression()
                    linreg.fit(data_set[flux + version].values.reshape(-1,1), y = data_set[flux + '_NT'+ ref].values.reshape(-1,1))
                    ax[0,1].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color="blue")

                    ax[0,1].text(data_set[flux + version].min(),data_set[flux + '_NT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_NT'+ ref], data_set[flux + version]):.2f}$")
                            
                    ax[0,1].set_xlim([data_set[flux + version].min()-1, data_set[flux + version].max()+1])
                    ax[0,1].set_ylim([data_set[flux + '_NT'+ ref].min()-1, data_set[flux + '_NT'+ ref].max()+1])
                    ax[0,1].set_xlabel(flux + version)
                    ax[0,1].set_ylabel(flux + '_NT'+ ref)

                            

                    figure3 = ax[0,2].scatter(data_set[flux + '_NT'+ ref], y = data_set[flux + '_DT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                    ax[0,2].plot([-25,25],[-25,25], linestyle = "--", color="black")
                    divider = make_axes_locatable(ax[0,2])
                    cax = divider.append_axes('bottom', size='5%', pad=0.7)
                    fig.colorbar(figure3, cax=cax, orientation='horizontal', label="doy")

                    linreg  = LinearRegression()
                    linreg.fit(data_set[flux + '_NT'+ ref].values.reshape(-1,1), y = data_set[flux + '_DT'+ ref].values.reshape(-1,1))
                    ax[0,2].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color = "blue")

                    ax[0,2].text(data_set[flux + '_NT'+ ref].min(),data_set[flux + '_DT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_DT'+ ref], data_set[flux + '_NT'+ ref]):.2f}$")           
                            
                    ax[0,2].set_xlim([data_set[flux + '_NT'+ ref].min()-1, data_set[flux + '_NT'+ ref].max()+1])
                    ax[0,2].set_ylim([data_set[flux + '_DT'+ ref].min()-1, data_set[flux + '_DT'+ ref].max()+1])
                    ax[0,2].set_xlabel(flux + '_NT'+ ref)
                    ax[0,2].set_ylabel(flux + '_DT'+ ref)
                    
                    
                    
                    figure1 = ax[1,0].scatter(data_set[flux + '_syn'], y = data_set[flux + '_DT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                    ax[1,0].plot([-25,25],[-25,25], linestyle = "--", color="black")
                    divider = make_axes_locatable(ax[1,0])
                    cax = divider.append_axes('bottom', size='5%', pad=0.7)
                    fig.colorbar(figure1, cax=cax, orientation='horizontal', label="doy")

                    linreg  = LinearRegression()
                    linreg.fit(data_set[flux + '_syn'].values.reshape(-1,1), y = data_set[flux + '_DT'+ ref].values.reshape(-1,1))
                    ax[1,0].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color="blue")

                    ax[1,0].text(data_set[flux + '_syn'].min(),data_set[flux + '_DT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_DT'+ ref], data_set[flux + '_syn']):.2f}$")

                    ax[1,0].set_xlim([data_set[flux + '_syn'].min()-1, data_set[flux + '_syn'].max()+1])
                    ax[1,0].set_ylim([data_set[flux + '_DT'+ ref].min()-1, data_set[flux + '_DT'+ ref].max()+1])
                    ax[1,0].set_xlabel(flux + '_syn')
                    ax[1,0].set_ylabel(flux + '_DT_' + ref)           
                            

                    figure2 = ax[1,1].scatter(data_set[flux + '_syn'], y = data_set[flux + '_NT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                    ax[1,1].plot([-25,25],[-25,25], linestyle = "--", color="black")
                    divider = make_axes_locatable(ax[1,1])
                    cax = divider.append_axes('bottom', size='5%', pad=0.7)
                    cbar = fig.colorbar(figure2, cax=cax, orientation='horizontal',label="doy")


                    linreg  = LinearRegression()
                    linreg.fit(data_set[flux + '_syn'].values.reshape(-1,1), y = data_set[flux + '_NT'+ ref].values.reshape(-1,1))
                    ax[1,1].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color="blue")

                    ax[1,1].text(data_set[flux + '_syn'].min(),data_set[flux + '_NT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_NT'+ ref], data_set[flux + '_syn']):.2f}$")
                            
                    ax[1,1].set_xlim([data_set[flux + '_syn'].min()-1, data_set[flux + '_syn'].max()+1])
                    ax[1,1].set_ylim([data_set[flux + '_NT'+ ref].min()-1, data_set[flux + '_NT'+ ref].max()+1])
                    ax[1,1].set_xlabel(flux + '_syn')
                    ax[1,1].set_ylabel(flux + '_NT'+ ref)

                            

                    figure3 = ax[1,2].scatter(data_set[flux + '_syn'], y = data_set[flux + '_orth'], c=data_set['doy'].values/1, cmap="twilight_shifted")
                    ax[1,2].plot([-25,25],[-25,25], linestyle = "--", color="black")
                    divider = make_axes_locatable(ax[1,2])
                    cax = divider.append_axes('bottom', size='5%', pad=0.7)
                    fig.colorbar(figure3, cax=cax, orientation='horizontal', label="doy")

                    linreg  = LinearRegression()
                    linreg.fit(data_set[flux + '_syn'].values.reshape(-1,1), y = data_set[flux + '_orth'].values.reshape(-1,1))
                    ax[1,2].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color = "blue")

                    ax[1,2].text(data_set[flux + '_syn'].min(),data_set[flux + '_orth'].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_orth'], data_set[flux + '_syn']):.2f}$")           
                            
                    ax[1,2].set_xlim([data_set[flux + '_syn'].min()-1, data_set[flux + '_syn'].max()+1])
                    ax[1,2].set_ylim([data_set[flux + '_orth'].min()-1, data_set[flux + '_orth'].max()+1])
                    ax[1,2].set_xlabel(flux + '_syn')
                    ax[1,2].set_ylabel(flux + '_orth')
                            
                    
                            
                    fig.suptitle(f"Comparison of different estimations of flux {flux} in {self.experiment_config['site']}")
                    fig.tight_layout()
                    fig.savefig(f'{self.PATH}/results/{data_type}/{flux}_{version}_scatter.png', bbox_inches='tight')   

                version = '_orth'
                fig, ax = plt.subplots(2,3, figsize=(20, 9))

                figure1 = ax[0,0].scatter(data_set[flux + version], y = data_set[flux + '_DT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                ax[0,0].plot([-25,25],[-25,25], linestyle = "--", color="black")
                divider = make_axes_locatable(ax[0,0])
                cax = divider.append_axes('bottom', size='5%', pad=0.7)
                fig.colorbar(figure1, cax=cax, orientation='horizontal', label="doy")

                linreg  = LinearRegression()
                linreg.fit(data_set[flux + version].values.reshape(-1,1), y = data_set[flux + '_DT'+ ref].values.reshape(-1,1))
                ax[0,0].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color="blue")

                ax[0,0].text(data_set[flux + version].min(),data_set[flux + '_DT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_DT'+ ref], data_set[flux + version]):.2f}$")

                ax[0,0].set_xlim([data_set[flux + version].min()-1, data_set[flux + version].max()+1])
                ax[0,0].set_ylim([data_set[flux + '_DT'+ ref].min()-1, data_set[flux + '_DT'+ ref].max()+1])
                ax[0,0].set_xlabel(flux + version)
                ax[0,0].set_ylabel(flux + '_DT_' + ref)           
                        

                figure2 = ax[0,1].scatter(data_set[flux + version], y = data_set[flux + '_NT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                ax[0,1].plot([-25,25],[-25,25], linestyle = "--", color="black")
                divider = make_axes_locatable(ax[0,1])
                cax = divider.append_axes('bottom', size='5%', pad=0.7)
                cbar = fig.colorbar(figure2, cax=cax, orientation='horizontal',label="doy")


                linreg  = LinearRegression()
                linreg.fit(data_set[flux + version].values.reshape(-1,1), y = data_set[flux + '_NT'+ ref].values.reshape(-1,1))
                ax[0,1].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color="blue")

                ax[0,1].text(data_set[flux + version].min(),data_set[flux + '_NT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_NT'+ ref], data_set[flux + version]):.2f}$")
                        
                ax[0,1].set_xlim([data_set[flux + version].min()-1, data_set[flux + version].max()+1])
                ax[0,1].set_ylim([data_set[flux + '_NT'+ ref].min()-1, data_set[flux + '_NT'+ ref].max()+1])
                ax[0,1].set_xlabel(flux + version)
                ax[0,1].set_ylabel(flux + '_NT'+ ref)

                        

                figure3 = ax[0,2].scatter(data_set[flux + '_NT'+ ref], y = data_set[flux + '_DT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                ax[0,2].plot([-25,25],[-25,25], linestyle = "--", color="black")
                divider = make_axes_locatable(ax[0,2])
                cax = divider.append_axes('bottom', size='5%', pad=0.7)
                fig.colorbar(figure3, cax=cax, orientation='horizontal', label="doy")

                linreg  = LinearRegression()
                linreg.fit(data_set[flux + '_NT'+ ref].values.reshape(-1,1), y = data_set[flux + '_DT'+ ref].values.reshape(-1,1))
                ax[0,2].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color = "blue")

                ax[0,2].text(data_set[flux + '_NT'+ ref].min(),data_set[flux + '_DT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_DT'+ ref], data_set[flux + '_NT'+ ref]):.2f}$")           
                        
                ax[0,2].set_xlim([data_set[flux + '_NT'+ ref].min()-1, data_set[flux + '_NT'+ ref].max()+1])
                ax[0,2].set_ylim([data_set[flux + '_DT'+ ref].min()-1, data_set[flux + '_DT'+ ref].max()+1])
                ax[0,2].set_xlabel(flux + '_NT'+ ref)
                ax[0,2].set_ylabel(flux + '_DT'+ ref)
                
                
                
                figure1 = ax[1,0].scatter(data_set[flux + '_syn'], y = data_set[flux + '_DT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                ax[1,0].plot([-25,25],[-25,25], linestyle = "--", color="black")
                divider = make_axes_locatable(ax[1,0])
                cax = divider.append_axes('bottom', size='5%', pad=0.7)
                fig.colorbar(figure1, cax=cax, orientation='horizontal', label="doy")

                linreg  = LinearRegression()
                linreg.fit(data_set[flux + '_syn'].values.reshape(-1,1), y = data_set[flux + '_DT'+ ref].values.reshape(-1,1))
                ax[1,0].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color="blue")

                ax[1,0].text(data_set[flux + '_syn'].min(),data_set[flux + '_DT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_DT'+ ref], data_set[flux + '_syn']):.2f}$")

                ax[1,0].set_xlim([data_set[flux + '_syn'].min()-1, data_set[flux + '_syn'].max()+1])
                ax[1,0].set_ylim([data_set[flux + '_DT'+ ref].min()-1, data_set[flux + '_DT'+ ref].max()+1])
                ax[1,0].set_xlabel(flux + '_syn')
                ax[1,0].set_ylabel(flux + '_DT_' + ref)           
                        

                figure2 = ax[1,1].scatter(data_set[flux + '_syn'], y = data_set[flux + '_NT'+ ref], c=data_set['doy'].values/1, cmap="twilight_shifted")
                ax[1,1].plot([-25,25],[-25,25], linestyle = "--", color="black")
                divider = make_axes_locatable(ax[1,1])
                cax = divider.append_axes('bottom', size='5%', pad=0.7)
                cbar = fig.colorbar(figure2, cax=cax, orientation='horizontal',label="doy")


                linreg  = LinearRegression()
                linreg.fit(data_set[flux + '_syn'].values.reshape(-1,1), y = data_set[flux + '_NT'+ ref].values.reshape(-1,1))
                ax[1,1].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color="blue")

                ax[1,1].text(data_set[flux + '_syn'].min(),data_set[flux + '_NT'+ ref].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_NT'+ ref], data_set[flux + '_syn']):.2f}$")
                        
                ax[1,1].set_xlim([data_set[flux + '_syn'].min()-1, data_set[flux + '_syn'].max()+1])
                ax[1,1].set_ylim([data_set[flux + '_NT'+ ref].min()-1, data_set[flux + '_NT'+ ref].max()+1])
                ax[1,1].set_xlabel(flux + '_syn')
                ax[1,1].set_ylabel(flux + '_NT'+ ref)

                        

                figure3 = ax[1,2].scatter(data_set[flux + '_syn'], y = data_set[flux + '_orth'], c=data_set['doy'].values/1, cmap="twilight_shifted")
                ax[1,2].plot([-25,25],[-25,25], linestyle = "--", color="black")
                divider = make_axes_locatable(ax[1,2])
                cax = divider.append_axes('bottom', size='5%', pad=0.7)
                fig.colorbar(figure3, cax=cax, orientation='horizontal', label="doy")

                linreg  = LinearRegression()
                linreg.fit(data_set[flux + '_syn'].values.reshape(-1,1), y = data_set[flux + '_orth'].values.reshape(-1,1))
                ax[1,2].plot([-25,25], [-25*linreg.coef_[0][0], 25*linreg.coef_[0][0]], color = "blue")

                ax[1,2].text(data_set[flux + '_syn'].min(),data_set[flux + '_orth'].max(), f"$y={linreg.coef_[0][0]: .2f}x + {linreg.intercept_[0]: .2f}$, $R^2 = {r2_score(data_set[flux + '_orth'], data_set[flux + '_syn']):.2f}$")           
                        
                ax[1,2].set_xlim([data_set[flux + '_syn'].min()-1, data_set[flux + '_syn'].max()+1])
                ax[1,2].set_ylim([data_set[flux + '_orth'].min()-1, data_set[flux + '_orth'].max()+1])
                ax[1,2].set_xlabel(flux + '_syn')
                ax[1,2].set_ylabel(flux + '_orth')
                        
                
                        
                fig.suptitle(f"Comparison of different estimations of flux {flux} in {self.experiment_config['site']}")
                fig.tight_layout()
                fig.savefig(f'{self.PATH}/results/{data_type}/{flux}_{version}_scatter.png', bbox_inches='tight')   
                    
        
            for flux, [data_type, data_set, time_window] in itertools.product(['GPP', 'RECO', 'NEE'], zip(['train', 'test'], 
                        [self.data_train, self.data_test], [self.experiment_config['train_period'], self.experiment_config['test_period']])):
                
                for year in range(*time_window):
                    fig, axes = plt.subplots(12,1, figsize=(30,40), sharex=True, sharey=True)

                    for month, ax in enumerate(axes.flatten()):
                        df_temp = data_set[(data_set["Month"] == month+1) & (data_set["Year"] == year)]
                        DT = df_temp[flux + "_DT"+ ref]
                        NT = df_temp[flux +"_NT"+ ref]
                        orth = df_temp[flux +"_orth"]
                        if flux == 'RECO':
                            orth_rest = df_temp[flux +"_orth_rest"]
                        
                        GT = df_temp[flux + '_syn']
                        
                        #if data_type == 'train':
                        #    nuisance = df_temp[flux +"_nuisance"]
                        
                        ax.plot(DT.values, color = "orange", label = "Daytime method")
                        ax.plot(NT.values, color = "black", label = "Nighttime method")
                        ax.plot(orth.values, color = "green", label = "Orthogonal ML")
                        ax.plot(GT.values, color = "yellow", label = "synthetic gronud truth")
                        if flux == 'RECO':
                            ax.plot(orth_rest.values, color = "blue", label = "Orthogonal ML rest")    
                        
                        #if data_type == 'train':
                        #    ax.plot(nuisance.values, color = "green", label = "Orthogonal nuisance")
                        if flux == 'NEE':
                            ax.set_ylim([-20, 20])
                        elif flux == 'GPP':
                            ax.set_ylim([-20, 30])
                        elif flux == 'RECO':
                            ax.set_ylim([-5, 15])


                        handles, labels = ax.get_legend_handles_labels()
                        fig.legend(handles, labels, loc=(0.45, 0.87))

                    fig.suptitle(f'Comparison of the predicted {flux} in monthly curves for different flux partitioning methods in {self.experiment_config["site"]} in {year}')
                    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
                    fig.savefig(f'{self.PATH}/results/{data_type}/monthly{flux}{year}.pdf', bbox_inches='tight')            
        
                
        return self
    

def clean_outliers(x, mean=True, times=2):
    if mean:
        crit = np.mean(x)
    else:
        crit = np.median(x)
    y = (np.roll(x,-1,0) + np.roll(x, 1, 0))/2
    conditions = [np.abs(x-y) < times*crit, np.abs(x - y) >= times*crit]
    choicelist = [x, y]
    x_new = np.select(conditions, choicelist)

    return x_new

def wdefcum(LE, P):
    n = len(LE)
    ET = LE / 2.45e6 * 1800
    wdefCum = np.zeros(n)
    wdefCum[1:] = np.NaN
    
    for i in range(1,n):
        wdefCum[i] = np.minimum(wdefCum[i-1] + P[i-1] - ET[i-1],0)
        
    if np.isnan(wdefCum[i]):
        wdefCum[i] = wdefCum[i-1]
    
    return wdefCum

def hyperbolic_transform(x, delta=10/1200, transform=False):   #alpha=10, beta=1400
    if not transform:
        return x
    else:
        return x/(x+(delta**-1))
       #return x/(delta*x+1)
    
    
def syntheticdataset(site, Q10=1.5, relnoise=0.0):
    folder_path = '../data/FLX_' + site + '/'
    file = 'FLX_' + site + '_FLUXNET2015_FULLSET_HH'    
    
    try:
        for file_names in os.listdir(folder_path):
            if file in file_names:
                data = pd.read_csv(folder_path + file_names)
    except FileNotFoundError as fnf_error:
        print(fnf_error)


    df = pd.read_csv('../data/Synthetic4BookChap.csv')
    SW_POT_sm_year = df.iloc[0:17520]['SW_POT_sm']
    SW_POT_sm_diff_year = df.iloc[0:17520]['SW_POT_sm_diff']
    
    times = len(data)//17520
    rest = len(data)- len(SW_POT_sm_year) * times
    
    SW_POT_sm = np.concatenate((np.tile(SW_POT_sm_year,(times)), SW_POT_sm[:rest]))
    SW_POT_sm_diff = np.concatenate((np.tile(SW_POT_sm_diff_year,(times)), SW_POT_sm_diff[:rest]))
    
    data['SW_POT_sm'] = SW_POT_sm
    data['SW_POT_sm_diff'] = SW_POT_sm_diff

    data_all = data.copy()       
    data = data.dropna()
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
    
    return data, data_all