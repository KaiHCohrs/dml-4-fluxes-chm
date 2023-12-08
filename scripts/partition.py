from argparse import ArgumentParser
from nneddyproc.utility.experiments import create_experiment_folder
from nneddyproc.training.trainers import trainer_basic
from nneddyproc.datasets.preprocessing import *
from nneddyproc.datasets.dataloader import *
from nneddyproc.models.models import *
from nneddyproc.analysis.visualization import monthly_curves, taylor_plot
import nneddyproc.analysis.postprocessing as post


def main(args):
    
    ################ Define the experiment  ################
    # Model
    model_config = {
                    'nn_lue': args.hidden_layer*[args.hidden_nodes],   # hidden layers without input
                    'nn_reco': args.hidden_layer*[args.hidden_nodes],
                    'ensemble_size': args.ensemble_size,
                    'nonlinearity': 'tanh'
                }
    
    # Data
    dataset_config = {'var_lue': ['SW_IN', 'SW_IN_POT', 'VPD', 'TA', 'SWC_1', 'WS', 'SW_POT_diff', 'SW_POT_sm','SW_POT_sm_diff', 'WD_cos', 'WD_sin', 'GPP_prox'],
                    'var_reco': ['TA', 'TS_1', 'SWC_1', 'WS', 'WD_cos', 'WD_sin', 'doy_sin', 'doy_cos', 'NEE_nt_avg','P_ERA', 'EF_dt_avg'], #, 
                    'test_portion': 0.3,
                    'batch_size': args.batch_size,
                    'seed': args.seed,
                    'site': args.site,
                    'year': args.year,
                    'quality_min': args.quality_min,
                    'target': 'NEE',
    }
    
    # Trainer
    trainer_config = { 'seed': args.seed,
                        'weight_decay': 0,
                        'patience': 7,
                        'lr_init': 0.001,
                        'max_iter': 400,
                        'tolerance': 1e-4,
                        'min_lr': 1e-7,
                        'lr_decay_factor': 0.5,
                        'lr_decay_steps': 7,
                        'track_training': True,
                        'device': 'cpu',
                        }
    
    ################ Save the experiment setup  ################    
    experiment_dict = {'model_config': model_config, 'data_config': dataset_config, 'trainer_config': trainer_config}

    #### Experiment ####
    print(f"Partitioning site-year: {args.site}-{args.year}")

    #### Create the experiment folder ####
    print("Creating the experiment folder...")
    experiment_path = create_experiment_folder(f"output_{args.site}_{args.year}", experiment_dict, path=args.results_folder)

    #### Prepare the data here ####
    print("Data processing in progress...")
    data = prepare_data(dataset_config, path=args.data_folder)
    
    
    #### Loop over the ensemble members ####
    fluxes = pd.DataFrame(columns=['NEE_QC'])
    fluxes['NEE_QC'] = data['NEE_QC']
    fluxes.index = data.index
    
    for i in range(args.ensemble_size):
        print(f"Start training model {i+1}/{args.ensemble_size}...")
        # sample a new seed
        dataset_config['seed'] = args.seed + i
        trainer_config['seed'] = args.seed + i
        
        train_dataloader, val_dataloader, scalers, available_vars = build_data_loaders(dataset_config, data)
        dataloaders = dict(
            train = train_dataloader,
            validation = val_dataloader
        )

        #### Run the actual experiment here ####
        model = LUEModel(*available_vars, model_config, scalers)
        outputs, model  = trainer_basic(model, dataloaders, **trainer_config)

        #### Create some analysis plots here ####
        X_lue = torch.tensor(data[available_vars[0]].values)
        SW_IN = torch.tensor(data['SW_IN'].values.reshape(-1,1))
        X_reco = torch.tensor(data[available_vars[1]].values)

        model = model.to('cpu')
        gpp, reco, nee, lue = model.predict(X_lue, SW_IN, X_reco)
    
        # store gpp, reco, nee in a csv file with quality flag
        fluxes[f'NEE_{i}'] = nee[:,0]
        fluxes[f'GPP_{i}'] = gpp[:,0]
        fluxes[f'RECO_{i}'] = reco[:,0]
        fluxes[f'LUE_{i}'] = lue[:,0]
    
        #store fluxes in the results folder
        fluxes.to_csv(experiment_path.joinpath("fluxes.csv"))
        print(f"Finish training model {i+1}/{args.ensemble_size}...")
        
        outputs_df = pd.DataFrame.from_dict(outputs)
        outputs_df.to_csv(experiment_path.joinpath(f"outputs_{i}.csv"))
        
    #store outputs dict also in a csv file\
    print(f"Generate plots of fluxes")
    
    NEE = [fluxes[f'NEE_{i}'] for i in range(args.ensemble_size)]
    GPP = [fluxes[f'GPP_{i}'] for i in range(args.ensemble_size)]
    RECO = [fluxes[f'RECO_{i}'] for i in range(args.ensemble_size)]
    LUE = [fluxes[f'LUE_{i}'] for i in range(args.ensemble_size)]
    
    # turn list of arrays into a dataframe
    NEE = pd.DataFrame(NEE).T
    GPP = pd.DataFrame(GPP).T
    RECO = pd.DataFrame(RECO).T
    LUE = pd.DataFrame(LUE).T
    
    GPP, RECO, NEE, LUE = post.prepare_data(GPP, RECO, NEE, LUE, ensemble_size=args.ensemble_size)
    NEE['NEE_QC'] = fluxes['NEE_QC']
    NEE['NEE_DT'] = -data['GPP_DT'] + data['RECO_DT']
    NEE['NEE_NT']  = -data['GPP_NT'] + data['RECO_NT']

    GPP['NEE_QC'] = fluxes['NEE_QC']
    GPP['GPP_DT'] = data['GPP_DT']
    GPP['GPP_NT']  = data['GPP_NT']
    
    RECO['NEE_QC'] = fluxes['NEE_QC']
    RECO['RECO_DT'] = data['RECO_DT']
    RECO['RECO_NT']  = data['RECO_NT']

    # plot the fluxes
    monthly_curves('NEE', NEE, results_path=experiment_path)
    monthly_curves('GPP', GPP, results_path=experiment_path)
    monthly_curves('RECO', RECO, results_path=experiment_path)
    #monthly_curves('LUE', LUE, results_path=experiment_path)
    
    # Make taylorplots
    taylor_plot(NEE, GPP, RECO, ensemble_size=args.ensemble_size, results_path=experiment_path)


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
