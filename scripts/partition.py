import os
import time
from argparse import ArgumentParser
import random as orandom
from pathlib import Path
import pandas as pd

#os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
os.environ["CUDA_VISIBLE_DEVICES"] = "" # 0 1 7
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".60"

import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stats

from dml4fluxes.experiments import experiment, experiment_utils
from dml4fluxes.analysis.visualization import monthly_curves, taylor_plot
from dml4fluxes.datasets.preprocessing import *
import dml4fluxes.analysis.postprocessing as post

def main(args):
    
    ################ Define the experiment  ################
    # Data
    dataset_config = {'X': ['VPD', 'TA', 'SWC_1', 'doy_sin','doy_cos'],
                    'W': [],
                    'var_reco': ['TA', 'TS_1', 'SWC_1', 'WS', 'WD_cos', 'WD_sin', 'doy_sin', 'doy_cos', 'NEE_nt_avg'], #, 'P_ERA', 'EF_dt_avg'
                    'test_portion': 0.3,
                    'batch_size': args.batch_size,
                    'seed': args.seed,
                    'site': args.site,
                    'year': args.year,
                    'quality_min': args.quality_min,
                    'target': 'NEE',
                    
                    ### deprecated and options, change for future
                    'site_name': args.site,
                    'syn': False,
                    'version': 'simple',
                    'Q10':1.5,
                    'relnoise':0,
                    'pre_computed':False,
                    'transform_T':True,
                    'month_wise': False,
                    'moving_window':[3, 5],
                    'delta': 'heuristic8',
                    'years': [args.year], # 'all',
                    'test_scenario': True,
                    'RMSE': True,
                    'alternative_fluxes': None,
                    'alternative_treatment': None, #"SW_IN_POT",
                    'good_years_only': True,
                    'store_parameter': True,
                    
                    
    }
    
    
    # Model
    model_y_config = dict(model = 'GradientBoostingRegressor',
                        min_samples_split = 5,
                        min_samples_leaf = 40,
                        max_depth=1,
                        n_estimators=300,
                        n_iter_no_change=5,
                        validation_fraction=0.3,
			            tol=0.01,
                        )

    model_t_config = dict(model = 'GradientBoostingRegressor',
                        min_samples_split = 5,
                        min_samples_leaf = 40,
                        max_depth=1,
                        n_estimators=300,
                        n_iter_no_change=5,
                        validation_fraction=0.3,
			            tol=0.01,
                        )
    model_lue_config = dict(model = 'GradientBoostingRegressor',
                        min_samples_split = 5,
                        min_samples_leaf = 40,
                        max_depth=1,
                        n_estimators=300,
                        n_iter_no_change=10,
                        validation_fraction=0.3,
			            tol=0.0001,
                        )
    
    model_reco_config = dict(
                    model = 'EnsembleCustomJNN',
                    model_config = {"layers": [len(dataset_config['var_reco']),15,15,1],
                                    "final_nonlin": True,
                                    "dropout_p": 0,
                                    "ensemble_size": 1},
                    trainer_config = {"weight_decay": 0.3, "iterations": 20000, "split": 1.0},
                    seed = 1
                    )
    
    dml_config = dict(cv=10)
    
    model_config = dict(y = model_y_config,
                    t = model_t_config,
                    lue= model_lue_config,
                    reco = model_reco_config,
                    dml = dml_config)
    
    ################ Save the experiment setup  ################    
    experiment_dict = {'model_config': model_config, 'data_config': dataset_config}

    #### Experiment ####
    print(f"Partitioning site-year: {args.site}-{args.year}")

    #### Create the experiment folder ####
    print("Creating the experiment folder...")
    exp = experiment.FluxPartDML(model_config, dataset_config)
    experiment_path = exp.new(args.site, args.year, experiment_dict, results_folder=args.results_folder)
    exp.prepare_data(path=args.data_folder)

    #### Loop over the ensemble members ####
    fluxes = pd.DataFrame(columns=['NEE_QC'])
    fluxes['NEE_QC'] = exp.data_all['NEE_QC']
    fluxes.index = exp.data_all.index
    
    for i in range(args.ensemble_size):

        print(f"Start training model {i+1}/{args.ensemble_size}...")
        dataset_config['seed'] = args.seed + i
        exp.model_config['reco']['seed'] = args.seed + i
        #### Run the actual experiment here ####
        exp.fit_models()
    
        # store gpp, reco, nee in a csv file with quality flag
        fluxes[f'NEE_{i}'] = exp.data_all['NEE_DML']
        fluxes[f'NEE_fit_{i}'] = exp.data_all['NEE_DML_fit']
        fluxes[f'GPP_{i}'] = exp.data_all['GPP_DML']
        fluxes[f'RECO_di_{i}'] = exp.data_all['RECO_DML_di']
        fluxes[f'RECO_res_{i}'] = exp.data_all['RECO_DML_res']
        fluxes[f'RECO_fit_{i}'] = exp.data_all['RECO_DML_fit']
        fluxes[f'LUE_{i}'] = exp.data_all['LUE_DML']
        
        #store fluxes in the results folder
        fluxes.to_csv(experiment_path.joinpath("fluxes.csv"))
        print(f"Finish training model {i+1}/{args.ensemble_size}...")
        
        #outputs_df = pd.DataFrame.from_dict(outputs)
        #outputs_df.to_csv(experiment_path.joinpath(f"outputs_{i}.csv"))
        
    #store outputs dict also in a csv file\
    # reread csv file
    fluxes = pd.read_csv(experiment_path.joinpath("fluxes.csv"), index_col=0)
    
    print(f"Generate plots of fluxes")
    
    NEE = [fluxes[f'NEE_{i}'] for i in range(args.ensemble_size)]
    NEE_fit = [fluxes[f'NEE_fit_{i}'] for i in range(args.ensemble_size)]
    GPP = [fluxes[f'GPP_{i}'] for i in range(args.ensemble_size)]
    RECO_fit = [fluxes[f'RECO_fit_{i}'] for i in range(args.ensemble_size)]
    # rename columns in RECO_fit to RECO_0, RECO_1, ...
    RECO_di = [fluxes[f'RECO_di_{i}'] for i in range(args.ensemble_size)]
    RECO_res = [fluxes[f'RECO_res_{i}'] for i in range(args.ensemble_size)]
    LUE = [fluxes[f'LUE_{i}'] for i in range(args.ensemble_size)]
    
    # turn list of arrays into a dataframe
    NEE = pd.DataFrame(NEE).T
    NEE_fit = pd.DataFrame(NEE_fit).T
    NEE_fit.columns = [f'NEE_{i}' for i in range(args.ensemble_size)]
    GPP = pd.DataFrame(GPP).T
    RECO_fit = pd.DataFrame(RECO_fit).T
    RECO_fit.columns = [f'RECO_{i}' for i in range(args.ensemble_size)]
    RECO_di = pd.DataFrame(RECO_di).T
    RECO_di.columns = [f'RECO_{i}' for i in range(args.ensemble_size)]
    RECO_res = pd.DataFrame(RECO_res).T
    RECO_res.columns = [f'RECO_{i}' for i in range(args.ensemble_size)]
    LUE = pd.DataFrame(LUE).T
    
    GPP, RECO_fit, NEE_fit, LUE = post.prepare_data(GPP, RECO_fit, NEE_fit, LUE, ensemble_size=args.ensemble_size)
    GPP, RECO_di, NEE, LUE = post.prepare_data(GPP, RECO_di, NEE, LUE, ensemble_size=args.ensemble_size)
    GPP, RECO_res, NEE, LUE = post.prepare_data(GPP, RECO_res, NEE, LUE, ensemble_size=args.ensemble_size)
    
    NEE['NEE_QC'] = fluxes['NEE_QC']
    NEE['NEE_DT'] = -exp.data_all['GPP_DT'] + exp.data_all['RECO_DT']
    NEE['NEE_NT']  = -exp.data_all['GPP_NT'] + exp.data_all['RECO_NT']

    NEE_fit['NEE_QC'] = fluxes['NEE_QC']
    NEE_fit['NEE_DT'] = -exp.data_all['GPP_DT'] + exp.data_all['RECO_DT']
    NEE_fit['NEE_NT']  = -exp.data_all['GPP_NT'] + exp.data_all['RECO_NT']

    GPP['NEE_QC'] = fluxes['NEE_QC']
    GPP['GPP_DT'] = exp.data_all['GPP_DT']
    GPP['GPP_NT']  = exp.data_all['GPP_NT']
    
    RECO_fit['NEE_QC'] = fluxes['NEE_QC']
    RECO_fit['RECO_DT'] = exp.data_all['RECO_DT']
    RECO_fit['RECO_NT']  = exp.data_all['RECO_NT']

    RECO_di['NEE_QC'] = fluxes['NEE_QC']
    RECO_di['RECO_DT'] = exp.data_all['RECO_DT']
    RECO_di['RECO_NT']  = exp.data_all['RECO_NT']

    RECO_res['NEE_QC'] = fluxes['NEE_QC']
    RECO_res['RECO_DT'] = exp.data_all['RECO_DT']
    RECO_res['RECO_NT']  = exp.data_all['RECO_NT']

    monthly_curves('NEE', NEE, results_path=experiment_path, suffix='_di')
    monthly_curves('NEE', NEE_fit, results_path=experiment_path, suffix='_fit')
    monthly_curves('GPP', GPP, results_path=experiment_path)
    monthly_curves('RECO', RECO_fit, results_path=experiment_path, suffix='_fit')
    monthly_curves('RECO', RECO_di, results_path=experiment_path, suffix='_di')
    monthly_curves('RECO', RECO_res, results_path=experiment_path, suffix='_res')
    #monthly_curves('LUE', LUE, results_path=experiment_path)

    taylor_plot(NEE, GPP, RECO_di, ensemble_size=args.ensemble_size, results_path=experiment_path, suffix='_di')
    taylor_plot(NEE_fit, GPP, RECO_fit, ensemble_size=args.ensemble_size, results_path=experiment_path, suffix='_fit')
    taylor_plot(NEE, GPP, RECO_res, ensemble_size=args.ensemble_size, results_path=experiment_path, suffix='_res')


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("-site", type=str, default="DE-Tha", help="Site of data to use")
    parser.add_argument("-year", type=int, default=2006, help="Year of data to use")
    parser.add_argument("-quality_min", type=int, default=0, help="minimum quality flag")
    parser.add_argument("-hidden_layer", type=int, default=2, help="hidden layer")
    parser.add_argument("-hidden_nodes", type=int, default=15, help="hidden nodes")
    parser.add_argument("-batch_size", type=int, default=256, help="batch size")
    parser.add_argument("-ensemble_size", type=int, default=1, help="number of partitioning runs")
    parser.add_argument("-seed", type=int, default=42, help="seed")
    parser.add_argument("-results_folder", type=str, default=None, help="Folder to save results")
    parser.add_argument("-data_folder", type=str, default=None, help="Folder to load data from")

    args = parser.parse_args()
    if args.data_folder is None:
        args.data_folder = pathlib.Path(__file__).parent.parent.joinpath('data')
    if args.results_folder is None:
        args.results_folder = pathlib.Path(__file__).parent.parent.joinpath('results')

    print(args.data_folder)
    print(args.results_folder)
    main(args)
