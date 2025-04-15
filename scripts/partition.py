from argparse import ArgumentParser
import pandas as pd
import pathlib

from dml4fluxes.experiments import experiment
from dml4fluxes.datasets.preprocessing import *

def main(args):
    # Define the experiment
    # Data
    dataset_config = {
        'X': ['VPD', 'TA', 'SWC_1', 'doy_sin', 'doy_cos'],
        'W': [],
        'var_reco': ['TA', 'TS_1', 'SWC_1', 'WS', 'WD_cos', 'WD_sin', 'doy_sin', 'doy_cos', 'NEE_nt_avg'],
        'test_portion': 0.3,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'site': args.site,
        'year': args.year,
        'quality_min': args.quality_min,
        'target': 'NEE',
        
        # Options for experimenting
        'site_name': args.site,
        'syn': False,
        'version': 'simple',
        'Q10': 1.5,
        'relnoise': 0,
        'pre_computed': False,
        'transform_T': True,
        'month_wise': False,
        'moving_window': [3, 5],
        'delta': 'heuristic8',
        'years': [args.year],
        'test_scenario': True,
        'RMSE': True,
        'alternative_fluxes': None,
        'alternative_treatment': None,
        'good_years_only': True,
        'store_parameter': True,
    }
    
    # Model
    model_y_config = {
        'model': 'GradientBoostingRegressor',
        'min_samples_split': 5,
        'min_samples_leaf': 40,
        'max_depth': 1,
        'n_estimators': 300,
        'n_iter_no_change': 5,
        'validation_fraction': 0.3,
        'tol': 0.01,
    }

    model_t_config = {
        'model': 'GradientBoostingRegressor',
        'min_samples_split': 5,
        'min_samples_leaf': 40,
        'max_depth': 1,
        'n_estimators': 300,
        'n_iter_no_change': 5,
        'validation_fraction': 0.3,
        'tol': 0.01,
    }
    
    model_lue_config = {
        'model': 'GradientBoostingRegressor',
        'min_samples_split': 5,
        'min_samples_leaf': 40,
        'max_depth': 1,
        'n_estimators': 300,
        'n_iter_no_change': 10,
        'validation_fraction': 0.3,
        'tol': 0.0001,
    }
    
    model_reco_config = {
        'model': 'EnsembleCustomJNN',
        'model_config': {
            "layers": [len(dataset_config['var_reco']), 15, 15, 1],
            "final_nonlin": True,
            "dropout_p": 0,
            "ensemble_size": 1
        },
        'trainer_config': {
            "weight_decay": 0.3, 
            "iterations": 20000, 
            "split": 1.0
        },
        'seed': 1
    }
    
    dml_config = {'cv': 10}
    
    model_config = {
        'y': model_y_config,
        't': model_t_config,
        'lue': model_lue_config,
        'reco': model_reco_config,
        'dml': dml_config
    }
    
    # Save the experiment setup
    experiment_dict = {'model_config': model_config, 'data_config': dataset_config}

    # Experiment
    print(f"Partitioning site-year: {args.site}-{args.year}")

    # Create the experiment folder
    print("Creating the experiment folder...")
    exp = experiment.FluxPartDML(model_config, dataset_config)
    experiment_path = exp.new(args.site, args.year, experiment_dict, results_folder=args.results_folder)
    exp.prepare_data(path=args.data_folder)

    # Loop over the ensemble members
    fluxes = pd.DataFrame(columns=['NEE_QC'])
    fluxes['NEE_QC'] = exp.data_all['NEE_QC']
    fluxes.index = exp.data_all.index
    
    for i in range(args.ensemble_size):
        print(f"Start training model {i+1}/{args.ensemble_size}...")
        dataset_config['seed'] = args.seed + i
        exp.model_config['reco']['seed'] = args.seed + i
        
        # Run the actual experiment
        exp.fit_models()
    
        # Store fluxes in a csv file with quality flag
        fluxes[f'NEE_{i}'] = exp.data_all['NEE_DML']
        fluxes[f'NEE_fit_{i}'] = exp.data_all['NEE_DML_fit']
        fluxes[f'GPP_{i}'] = exp.data_all['GPP_DML']
        fluxes[f'RECO_di_{i}'] = exp.data_all['RECO_DML_di']
        fluxes[f'RECO_res_{i}'] = exp.data_all['RECO_DML_res']
        fluxes[f'RECO_fit_{i}'] = exp.data_all['RECO_DML_fit']
        fluxes[f'LUE_{i}'] = exp.data_all['LUE_DML']
        
        # Store fluxes in the results folder
        fluxes.to_csv(experiment_path.joinpath("fluxes.csv"))
        print(f"Finish training model {i+1}/{args.ensemble_size}...")
    
    # Store outputs dict also in a csv file
    fluxes = pd.read_csv(experiment_path.joinpath("fluxes.csv"), index_col=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--site", type=str, default="DE-Tha", help="Site of data to use")
    parser.add_argument("--year", type=int, default=2006, help="Year of data to use")
    parser.add_argument("--quality_min", type=int, default=0, help="minimum quality flag")
    parser.add_argument("--hidden_layer", type=int, default=2, help="hidden layer")
    parser.add_argument("--hidden_nodes", type=int, default=15, help="hidden nodes")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--ensemble_size", type=int, default=1, help="number of partitioning runs")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--results_folder", type=str, default=None, help="Folder to save results")
    parser.add_argument("--data_folder", type=str, default=None, help="Folder to load data from")

    args = parser.parse_args()
    if args.data_folder is None:
        args.data_folder = pathlib.Path(__file__).parent.parent.joinpath('data')
    if args.results_folder is None:
        args.results_folder = pathlib.Path(__file__).parent.parent.joinpath('results')

    print(args.data_folder)
    print(args.results_folder)
    main(args)
