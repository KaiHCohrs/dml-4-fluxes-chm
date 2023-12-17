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

from dml4fluxes.datasets.preprocessing import load_data, unwrap_time, standardize_column_names,\
                                                sw_pot_sm, sw_pot_sm_diff,\
                                                diffuse_to_direct_rad, NEE_quality_masks,\
                                                quality_check, GPP_prox,\
                                                normalize, wdefcum, check_available_variables,\
                                                make_cyclic, sw_pot_diff, prepare_data
from dml4fluxes.datasets.generate_data import synthetic_dataset
                                                
from dml4fluxes.analysis.postprocessing import evaluate, timely_averages, condense
from dml4fluxes.models import models
from .utility import get_available_sites, get_igbp_of_site, transform_t, JSONEncoder, create_experiment_folder

class FluxPartDML():
    
    def __init__(self, model_config, dataset_config):
        self.PATH = None
        self.dataset_config = dataset_config
        self.model_config = model_config
                
    def new(self, site, year, experiment_dict, results_folder):
        #Start a new experiment.
        self.PATH = create_experiment_folder(f"output_{site}_{year}", experiment_dict, path=results_folder)
        return self.PATH

    def prepare_data(self, path=None):
        self.data = prepare_data(self.dataset_config, path=path)
        
        # Setting another GT flux (could be DT, NT or NEE for instance)
        if self.dataset_config['alternative_fluxes']:
            self.data['NEE'] = self.data['NEE_' + self.dataset_config['alternative_fluxes']]
            self.data['RECO'] = self.data['RECO_' + self.dataset_config['alternative_fluxes']]
            self.data['GPP'] = self.data['GPP_' + self.dataset_config['alternative_fluxes']]
        if self.dataset_config['alternative_treatment']:
            self.data['SW_IN'] = self.data[self.dataset_config['alternative_treatment']]
        
        # Reduce variables to the ones also available
        print('Check inputs for W')
        self.W_var = check_available_variables(self.dataset_config['W'], self.data.columns, self.data)
        print('Check inputs for X')
        self.X_var = check_available_variables(self.dataset_config['X'], self.data.columns, self.data)
        print('Check inputs for RECO')
        self.RECO_var = check_available_variables(self.dataset_config['var_reco'], self.data.columns, self.data)
        self.model_config['reco']['model_config']['layers'][0] = len(self.RECO_var)

        # Generate mask for quality of all data to be used.
        self.data['QC'] = quality_check(self.data, self.X_var + self.W_var + ['SW_IN'])
        
        # Filter by quality mask
        N = len(self.data)
        self.data_all = self.data.copy()
        self.data = self.data.loc[self.data['QM_halfhourly'].astype(bool),:]        
        self.data = self.data.dropna(subset=self.W_var+self.X_var+self.RECO_var+ ['SW_IN'] +['NEE'], how='any') 
    
        # Print ratio of filtered data
        print(f'Ratio of filtered data: {len(self.data)/N}')
        
        # Or previous year for the last one.
        for var in self.X_var + self.W_var + self.RECO_var + ['SW_IN']:
            if var.endswith('_n') or var.endswith('_s'):
                self.data = normalize(self.data, var[:-2], norm_type=var[-1])  
            else:
                pass

    def fit_models(self):
        # Run the fitting and partitioning
        self.fitted_models = []
        if self.dataset_config['syn']:
            self.data['NEE_QC'] = 0
            #TODO: Not sure if relevant
            self.data['QC']=0

        self.data['T'] = self.data['SW_IN']
        self.data['GPP_DML'] = 0
        self.data['RECO_DML'] = 0
        self.data['NEE_DML'] = 0
        self.data['T_DML'] = 0
        self.data['woy'] = -99
            
        self.data.loc[:, 'GPP_DML'] = None
        self.data.loc[:, 'RECO_DML'] = None
        self.data.loc[:, 'RECO_DML_res'] = None
        self.data.loc[:, 'NEE_DML'] =  None
        self.data.loc[:, 'T_DML'] = None
        self.data.loc[:, 'LUE_DML'] = None
                
        mask = (self.data['NEE_QC']==0)             
        target = 'NEE'

        self.data['T'], parameter = transform_t(x=self.data['SW_IN'],
                                                    delta=self.dataset_config['delta'],
                                                    data=self.data,
                                                    month_wise=self.dataset_config['month_wise'],
                                                    moving_window=self.dataset_config['moving_window'],
                                                    target = target,
                                                    )
        self.parameter = parameter
        self.T = self.data.loc[mask, 'T'].values
        self.X = self.data.loc[mask, self.X_var].values
            
        if not len(self.W_var):
            self.W = None
        else:
            self.W = self.data.loc[mask, self.W_var].values

        if not len(self.RECO_var):
            self.X_reco = None
        else:
            self.X_reco = self.data.loc[mask, self.RECO_var].values

        self.Y = self.data.loc[mask, target].values

        self.dml = models.dml_fluxes(self.model_config['y'],
                                    self.model_config['t'],
                                    self.model_config['lue'],
                                    self.model_config['dml'],
                                    self.model_config['reco']
                                    )
        self.dml.fit(self.Y,self.X,self.T,self.W, self.X_reco)
            
        self.dml.score_train = self.dml.get_score(self.X, self.T, self.Y, self.W)

        self.light_response_curve = models.LightResponseCurve(self.dataset_config['moving_window'],
                                                            self.parameter, 
                                                            self.dataset_config['delta'], 
                                                            self.dml.lue)
        self.fitted_models.append(self.dml)

        self.data_all['T'], alphas, betas = transform_t(x=self.data_all['SW_IN'],
                                                    delta=self.dataset_config['delta'],
                                                    data=self.data_all,
                                                    month_wise=self.dataset_config['month_wise'],
                                                    moving_window=self.dataset_config['moving_window'],
                                                    parameter = parameter)
        
        self.data_all = self.data_all.dropna(subset=['T']+self.X_var+self.W_var+['SW_IN'], how='any')
        T = self.data_all['T'].values
        X = self.data_all[self.X_var].values
    
        if self.dataset_config['transform_T']:
            self.data_all["alpha"] = alphas
            self.data_all["beta"] = betas
            
        if not len(self.W_var):
            W = None
        else:
            W = self.data_all[self.W_var].values

        Y = self.data_all['NEE'].values

        self.data_all = self.data_all.dropna(subset=self.RECO_var, how='any')
        if not len(self.RECO_var):
            X_reco = None
        else:
            X_reco = self.data_all[self.RECO_var].values


        self.data_all['GPP_DML'] = self.dml.gpp(X, T)
        self.data_all['RECO_DML_di'] = self.dml.reco(X, T, W)
        if 'RECO_DML_res' not in self.data_all.columns:
            self.data_all['RECO_DML_res'] = None
        self.data_all['RECO_DML_res'] = self.dml.reco_res(Y, X, T)
        self.data_all['NEE_DML'] = self.dml.nee(X, T, W)
        self.data_all['T_DML'] = self.dml.t(X, W)
        if 'LUE_DML' not in self.data_all.columns:
            self.data_all['LUE_DML'] = None
        self.data_all['LUE_DML'] = self.dml.lue(X)
        self.data_all['RECO_DML_fit'] = self.dml.model_reco.predict(X_reco)
        self.data_all['NEE_DML_fit'] = -self.data_all['GPP_DML'] + self.data_all['RECO_DML_fit']

        return None