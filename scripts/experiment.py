import os
import time
from argparse import ArgumentParser
import random as orandom
from pathlib import Path
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stats

from dml4fluxes.experiments import experiment, experiment_utils
from dml4fluxes.analysis.visualization import *
from dml4fluxes.datasets.preprocessing import *
from dml4fluxes.analysis.postprocessing import *


def main(parser: ArgumentParser = None, **kwargs):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("-site", type=str, default='all', help="Sites on which to run the partitioning.")
    parser.add_argument('-syn', action='store_true')
    parser.add_argument('-no-syn', dest='syn', action='store_false')
    parser.add_argument("-relnoise", type=float, default=0, help="Relative noise on synthetic data.")
    parser.add_argument('-nonlin-t', action='store_true')
    parser.add_argument('-lin-t', dest='nonlin_t', action='store_false')
    parser.add_argument('-max_depth', type=int, default=1, help='Depth of gradient boosting regressors.')
    parser.add_argument('-X', type=int, default=1, help='Selection of inputs.')
    
    time.sleep(orandom.uniform(1, 20))

    args = parser.parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)
    
    print('>>> Starting experiment.')
    
    #results_path = Path(__file__).parent.parent.joinpath('results')
    #csv_name = f"bayesQ10_{args.samples}_{args.method}_{args.ml}_{dropout}_{args.number}.csv"

    if args.X == 1:
        X = ['VPD', 'TA', 'doy_sin','doy_cos']
    elif args.X == 2:
        X = ['VPD', 'TA', 'SWC_1', 'doy_sin','doy_cos']
    elif args.X == 3:
        X = ['VPD', 'TA', 'SW_ratio', 'doy_sin','doy_cos']
    elif args.X == 4:
        X = ['VPD', 'TA', 'SWC_1', 'SW_ratio', 'doy_sin','doy_cos']
    elif args.X == 5:
        X = ['VPD', 'TA', 'CWD', 'SW_ratio', 'doy_sin','doy_cos']
    elif args.X == 6:
        X = ['VPD', 'TA', 'CWD', 'doy_sin','doy_cos']


    pre_computed=False
    syn=args.syn
    relnoise=0
    site = str(args.site)
    dataset_config = dict(
                site_name = site,
                syn = syn,
                version = 'simple',
                Q10=1.5,
                relnoise=args.relnoise,
                pre_computed=pre_computed,
                transform_T = args.nonlin_t,
                month_wise= False,
                moving_window=[3, 5],
                delta = 'heuristic8',
                X = X, #'CWD'
                W = [],
                years = 'all', # 'all',
                test_scenario=False,
                RMSE=True,
                alternative_fluxes=None,
                alternative_treatment=None, #"SW_IN_POT",
                good_years_only=True,
                store_parameter=True,
                )

    model_y_config = dict(model = 'GradientBoostingRegressor',
                        min_samples_split = 5,
                        min_samples_leaf = 40,
                        max_depth=args.max_depth,
                        n_estimators=300,
                        n_iter_no_change=5,
                        validation_fraction=0.3,
			tol=0.01,
			)

    model_t_config = dict(model = 'GradientBoostingRegressor',
                        min_samples_split = 5,
                        min_samples_leaf = 40,
                        max_depth=args.max_depth,
                        n_estimators=300,
                        n_iter_no_change=5,
                        validation_fraction=0.3,
			tol=0.01,
                        )
    model_final_config = dict(model = 'GradientBoostingRegressor',
                        min_samples_split = 5,
                        min_samples_leaf = 40,
                        max_depth=args.max_depth,
                        n_estimators=300,
                        n_iter_no_change=10,
                        validation_fraction=0.3,                        
			tol=0.00001,
			)
    
    dml_config = dict(cv=10)

    model_configs = dict(y = model_y_config,
                        t = model_t_config,
                        final= model_final_config,
                        dml =dml_config)

    experiment_config = dict(experiment_type = 'flux_partitioning',
                            comment='Flux partitioning',
                            extrapolate=False,
                            seed=1000)

    exp = experiment.FluxPartDML()
    exp.new(site)
    exp.configs(experiment_config, dataset_config, model_configs)
    
    if args.site == 'all':
        exp.all_partitions()
    else:
        exp.prepare_data()
        exp.fit_models()
    
    
    tables, dictionary = numerical_cross_consistency(exp.experiment_name, syn=False)
    tables, dictionary = numerical_cross_consistency(exp.experiment_name, syn=False, partial='DT')
    tables, dictionary = numerical_cross_consistency(exp.experiment_name, syn=False, partial='NT')

    del exp


if __name__ == '__main__':
    main()
