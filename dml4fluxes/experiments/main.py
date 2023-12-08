import os
import json
import sys
import time
import subprocess


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


path = os.path.join(os.path.dirname(__file__), "../../")
sys.path.insert(1, path)


def main():
    print(os.getcwd())
    path = os.path.join(os.path.dirname(__file__), "../../exp/")
    os.chdir(path)

    # For runnning it on a cluster if not installed in the singularity container yet
    install("doubleml")
    install("econml")

    from dml4fluxes.experiments import experiment

    experiment_config_file = "experiment_files/" + str(sys.argv[1])
    with open(experiment_config_file) as json_file:
        experiment_config = json.load(json_file)

    if experiment_config["experiment_type"] == "flux_partitioning":
        dataset_config_file = experiment_config["dataset_config"]
        model_config_file = experiment_config["model_config"]

        experiment_type = experiment_config["experiment_type"]
        seed = experiment_config["seed"]
        with open("experiment_files/" + model_config_file) as json_file:
            model_config = json.load(json_file)
        with open("experiment_files/" + dataset_config_file) as json_file:
            dataset_config = json.load(json_file)

        start_time = time.time()
        exp = experiment.FluxPartDML2()
        exp.new("all")
        exp.configs(experiment_config, dataset_config, model_config)
        exp.all_analysis()
        print(time.time() - start_time)
        print("done")


if __name__ == "__main__":
    main()
