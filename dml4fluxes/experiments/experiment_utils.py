import os
import json
import copy
import csv
import math
import itertools
import numpy as np

from sklearn.utils import check_random_state


def build_single_experiment(experiment_config, model_config, dataset_config):
    if experiment_config["experiment_type"] == "flux_partitioning":

        PATH = os.path.join(os.path.dirname(__file__), "../../exp/")
        i = 1
        while os.path.exists(
            PATH + "experiment_files/experiment_config{}.txt".format(i)
        ):
            i += 1

        experiment_config["dataset_config"] = "dataset_config{}.txt".format(i)
        experiment_config["model_config"] = "model_config{}.txt".format(i)

        with open(
            PATH + "experiment_files/experiment_config{}.txt".format(i), "w"
        ) as outfile:
            json.dump(experiment_config, outfile)
        with open(
            PATH + "experiment_files/dataset_config{}.txt".format(i), "w"
        ) as outfile:
            json.dump(dataset_config, outfile)
        with open(
            PATH + "experiment_files/model_config{}.txt".format(i), "w"
        ) as outfile:
            json.dump(model_config, outfile)
        print("Congratulations! Experiment{}".format(i) + " is ready!")
