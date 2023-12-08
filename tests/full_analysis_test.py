import shutil
import sys
sys.path.insert(1, '../../dml4fluxes/')

from dml4fluxes.experiments import experiment

def single_site_test():
    print('Test a single site over some years on real data with transformation.')
    dataset_config = dict(
                site_name = "FI-Hyy",
                syn = False,
                version = 'simple',
                Q10=1.5,
                relnoise=0,
                transform_T = True,
                month_wise= True,
                delta = 'heuristic3', #fixed value, max_heuristic, slope_heuristic 
                X = ['VPD', 'TA','wdefcum', 'SW_POT_sm', 'SW_POT_sm_diff'],
                #X = ['VPD', 'TA','wdefcum', 'SW_POT_sm', 'SW_POT_sm_diff', 'PA', 'WS', 'GPP_prox', 'P', 'WD_sin', 'WD_cos', 'NEE_nt_avg', 'TS_1', 'TS_2', 'SWC_1', 'SWC_2', 'SW_DIF'], #['VPD', 'TA'],#
                #X = ['VPD', 'TA', 'TS_1', 'TS_2', 'SWC_1', 'SWC_2'],
                W = [], #['SW_POT_sm', 'SW_POT_sm_diff'], #['wdefcum'],
                years = [2012,2013,2014], # 'all'
                norm_type = 'standardize'
                )
        
    model_y_config = dict(model = 'GradientBoostingRegressor',
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
                        max_depth=1,
                        n_estimators=300)
                        #learning_rate=0.1,
                        #max_depth=3
    dml_config = dict(cv=10)

    model_configs = dict(y = model_y_config,
                        t = model_t_config,
                        final= model_final_config,
                        dml =dml_config)

    experiment_config = dict(experiment_type = "flux_partitioning",
                            comment='Test',
                            extrapolate=False,
                            seed=1000)

    exp = experiment.FluxPartDML()
    exp.new('FI-Hyy')
    exp.configs(experiment_config, dataset_config, model_configs)
    exp.prepare_data()
    exp.fit_models()
    #exp.all_analysis()
    dir_path = exp.PATH

    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))


def single_site_test_gen_data():
    print('Test a single site over some years on generated data without.')
    dataset_config = dict(
                site_name = "FI-Hyy",
                syn = True,
                version = 'simple',
                Q10=1.5,
                relnoise=0.1,
                pre_computed=True,
                transform_T = False,
                month_wise= True,
                delta = 'heuristic3', #fixed value, max_heuristic, slope_heuristic 
                X = ['VPD', 'TA'],
                #X = ['VPD', 'TA','wdefcum', 'SW_POT_sm', 'SW_POT_sm_diff', 'PA', 'WS', 'GPP_prox', 'P', 'WD_sin', 'WD_cos', 'NEE_nt_avg', 'TS_1', 'TS_2', 'SWC_1', 'SWC_2', 'SW_DIF'], #['VPD', 'TA'],#
                #X = ['VPD', 'TA', 'TS_1', 'TS_2', 'SWC_1', 'SWC_2'],
                W = ['SW_POT_sm', 'SW_POT_sm_diff'], #['SW_POT_sm', 'SW_POT_sm_diff'], #['wdefcum'],
                years = 'all', # 'all'
                norm_type = 'standardize'
                )
        
    model_y_config = dict(model = 'GradientBoostingRegressor',
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
                        max_depth=1,
                        n_estimators=300)
                        #learning_rate=0.1,
                        #max_depth=3
    dml_config = dict(cv=10)

    model_configs = dict(y = model_y_config,
                        t = model_t_config,
                        final= model_final_config,
                        dml =dml_config)

    experiment_config = dict(experiment_type = "flux_partitioning",
                            comment='Test',
                            extrapolate=False,
                            seed=1000)

    exp = experiment.FluxPartDML()
    exp.new('FI-Hyy')
    exp.configs(experiment_config, dataset_config, model_configs)
    exp.prepare_data()
    exp.fit_models()
    #exp.all_analysis()
    dir_path = exp.PATH

    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))
    
    
def all_sites_test():
    print('Test on all sites over the last year on real data with transformation.')
    dataset_config = dict(
                site_name = "all",
                syn = False,
                version = 'simple',
                Q10=1.5,
                relnoise=0,
                transform_T = True,
                month_wise= True,
                delta = 'heuristic3', #fixed value, max_heuristic, slope_heuristic 
                X = ['VPD', 'TA','wdefcum', 'SW_POT_sm', 'SW_POT_sm_diff'],
                #X = ['VPD', 'TA','wdefcum', 'SW_POT_sm', 'SW_POT_sm_diff', 'PA', 'WS', 'GPP_prox', 'P', 'WD_sin', 'WD_cos', 'NEE_nt_avg', 'TS_1', 'TS_2', 'SWC_1', 'SWC_2', 'SW_DIF'], #['VPD', 'TA'],#
                #X = ['VPD', 'TA', 'TS_1', 'TS_2', 'SWC_1', 'SWC_2'],
                W = [], #['SW_POT_sm', 'SW_POT_sm_diff'], #['wdefcum'],
                years = 'last', # 'all'
                norm_type = 'standardize'
                )
        
    model_y_config = dict(model = 'GradientBoostingRegressor',
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
                        max_depth=1,
                        n_estimators=300)
                        #learning_rate=0.1,
                        #max_depth=3
    dml_config = dict(cv=10)

    model_configs = dict(y = model_y_config,
                        t = model_t_config,
                        final= model_final_config,
                        dml =dml_config)

    experiment_config = dict(experiment_type = "flux_partitioning",
                            comment='Test',
                            extrapolate=False,
                            seed=1000)

    exp = experiment.FluxPartDML()
    exp.new('all')
    exp.configs(experiment_config, dataset_config, model_configs)
    #exp.prepare_data()
    #exp.fit_models()
    exp.all_analysis()
    dir_path = exp.PATH

    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))    


def all_sites_gen_test():
    print('Test on all sites over the last year on real data with transformation.')
    dataset_config = dict(
                site_name = "all",
                syn = True,
                version = 'simple',
                Q10=1.5,
                relnoise=0.1,
                pre_computed=True,
                transform_T = False,
                month_wise= True,
                delta = 'heuristic3', #fixed value, max_heuristic, slope_heuristic 
                X = ['VPD', 'TA'],
                #X = ['VPD', 'TA','wdefcum', 'SW_POT_sm', 'SW_POT_sm_diff', 'PA', 'WS', 'GPP_prox', 'P', 'WD_sin', 'WD_cos', 'NEE_nt_avg', 'TS_1', 'TS_2', 'SWC_1', 'SWC_2', 'SW_DIF'], #['VPD', 'TA'],#
                #X = ['VPD', 'TA', 'TS_1', 'TS_2', 'SWC_1', 'SWC_2'],
                W = ['SW_POT_sm', 'SW_POT_sm_diff'], #['SW_POT_sm', 'SW_POT_sm_diff'], #['wdefcum'],
                years = 'all', # 'all'
                norm_type = 'standardize'
                )
        
    model_y_config = dict(model = 'GradientBoostingRegressor',
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
                        max_depth=1,
                        n_estimators=300)
                        #learning_rate=0.1,
                        #max_depth=3
    dml_config = dict(cv=10)

    model_configs = dict(y = model_y_config,
                        t = model_t_config,
                        final= model_final_config,
                        dml =dml_config)

    experiment_config = dict(experiment_type = "flux_partitioning",
                            comment='Test',
                            extrapolate=False,
                            seed=1000)

    exp = experiment.FluxPartDML()
    exp.new('all')
    exp.configs(experiment_config, dataset_config, model_configs)
    #exp.prepare_data()
    #exp.fit_models()
    exp.all_analysis()
    dir_path = exp.PATH

    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))    


if __name__ == '__main__':
    #assert single_site_test()
    #assert single_site_test_gen_data()
    #assert all_sites_test()
    assert all_sites_gen_test()
    print("Tests passed")