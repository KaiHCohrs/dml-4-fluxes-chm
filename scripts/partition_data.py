import shutil
import sys
import argparse
sys.path.insert(1, '../../dml4fluxes/')

from dml4fluxes.experiments import experiment

def main(pre_computed, syn, relnoise, Q10_DML):

    Q10_fitting=False    
    for noise in relnoise:
        if noise == 0.0:
            noise = 0
        if syn:
            X = ['VPD', 'TA']
            W = ['SW_POT_sm', 'SW_POT_sm_diff']
        else:
            X = ['VPD', 'TA', 'wdefcum', 'SW_POT_sm', 'SW_POT_sm_diff']
            W = []
        if Q10_DML=="None":
            model_y = 'GradientBoostingRegressor'
        elif Q10_DML=="pre":
            model_y = 'Reco_DML'
        else:
            model_y = 'GradientBoostingRegressor'
            Q10_fitting = True
        
        dataset_config = dict(
                    site_name = "all",
                    syn = syn,
                    version = 'simple',
                    Q10=1.5,
                    relnoise=relnoise,
                    pre_computed=pre_computed,
                    transform_T = not syn,
                    month_wise= False,
                    moving_window=[3, 3],
                    delta = 'heuristic3', #fixed value, max_heuristic, slope_heuristic 
                    X = X,
                    #X = ['VPD', 'TA','wdefcum', 'SW_POT_sm', 'SW_POT_sm_diff', 'PA', 'WS', 'GPP_prox', 'P', 'WD_sin', 'WD_cos', 'NEE_nt_avg', 'TS_1', 'TS_2', 'SWC_1', 'SWC_2', 'SW_DIF'], #['VPD', 'TA'],#
                    #X = ['VPD', 'TA', 'TS_1', 'TS_2', 'SWC_1', 'SWC_2'],
                    W = W, #['SW_POT_sm', 'SW_POT_sm_diff'], #['wdefcum'],
                    Q10_drivers = ['SW_IN'],
                    years = 'all', # 'all'
                    norm_type = 'standardize',
                    Q10_fitting=False,
                    Q10_include_SW=False,
                    RMSE=True,
                    test_scenario=False,
                    alternative_fluxes='ann',
                    )
            
        #model_y_config = dict(model = 'GradientBoostingRegressor',
        #                    min_samples_split = 5,
        #                    min_samples_leaf = 40,
        #                    max_depth=1,
        #                    n_estimators=300,
        #                    #learning_rate=0.1,
        #                    #max_depth=3
        #                    )

        model_y_config = dict(model = model_y,
                            min_samples_split = 5,
                            min_samples_leaf = 40,
                            max_depth=1,
                            n_estimators=300,
                            #learning_rate=0.1,
                            #max_depth=3
                            )

        model_t_config = dict(model = 'GradientBoostingRegressor',
                            min_samples_split = 5,
                            min_samples_leaf = 40,
                            max_depth=1,
                            n_estimators=300,
                            # max_depth=3
                            )
        model_final_config = dict(model = 'GradientBoostingRegressor',
                            min_samples_split = 5,
                            min_samples_leaf = 40,
                            max_depth=2,
                            n_estimators=300)
                            #learning_rate=0.1,
                            #max_depth=3
        dml_config = dict(cv=10)

        model_configs = dict(y = model_y_config,
                            t = model_t_config,
                            final= model_final_config,
                            dml =dml_config)


        experiment_config = dict(experiment_type = 'flux_partitioning',
                                comment='Flux partitioning',
                                extrapolate=False,
                                seed=1000)

        print(f"Start noise level {noise}.")
        exp = experiment.FluxPartDML()
        exp.new('all')
        exp.configs(experiment_config, dataset_config, model_configs)
        #exp.prepare_data()
        #exp.fit_models()
        exp.all_analysis()
        print(f"Finished noise level {noise}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run flux partitioning "\
                                    "on real or generated data for certain noise level.")

    parser.add_argument('--pre_computed', dest='pre_computed', default=False, action='store_true',
                        help="If flag is set, the values are taken from the precomputed datasets.")
    parser.add_argument('--syn', dest='syn', default=False, action='store_true',
                        help="If flag is set, the data is generated similar to the bookchapter.")
    parser.add_argument("--relnoise", dest='relnoise', type=float, nargs="+", required=False, default=[0],
                        help="Amount of relative noise for generating the data. Only relevant for "\
                        "synthetic data.")
    parser.add_argument("--Q10_DML", dest='Q10_DML', type=str, required=False, default="None",
                    help="Specify if with None, pre, post if no night-time or on the residuals "\
                        "a nonparametric Q10 model should be fitted.")
    args = parser.parse_args()
    main(args.pre_computed, args.syn, args.relnoise, args.Q10_DML)