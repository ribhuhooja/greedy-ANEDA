import logging
import os.path
from datetime import datetime
from testing_functions import run_some_linear_models, run_nn

from data_helper import read_yaml
from datasets_generator import create_train_val_test_sets
from Trainer import Trainer
from utils import make_log_folder, generate_config_list
import time
from neural_net.NeuralNet1 import NeuralNet1

if __name__ == '__main__':
    ## Read the config file:
    data_generator_config = read_yaml("../configs/data_generator.yaml")

    ## Make a log folder to log the experiments
    make_log_folder(log_folder_name=data_generator_config["log_path"])

    # Make a list of configs, we'll be running the model for each config
    config_list = generate_config_list(data_generator_config)

    for i, config in enumerate(config_list):
        t_start = time.time()
        logging.basicConfig(filename=os.path.join(config["log_path"], config["data"]["file_name"] + ".log"),
                            level=logging.INFO)
        logging.info("Start: " + str(i + 1) + " - " + config["data"]["file_name"] + " at " + datetime.now().strftime(
            "%m/%d/%Y %H:%M:%S "))

        dataset = create_train_val_test_sets(config)
        params_net1 = read_yaml("../configs/neural_net_1.yaml")

        ## Train model
        val_metrics = Trainer.train_model(NeuralNet1, dataset, params_net1)

        logging.info(val_metrics)

        t_end = time.time()
        logging.info(
            f"End: in {round((t_end - t_start) / 60, 2)} mins, - " + str(
                i + 1) + "th config - " + datetime.now().strftime(
                "%m/%d/%Y %H:%M:%S "))
        logging.info("\n------------\n")

    #### Old code
    # ## Read the config file:
    # data_generator_config = read_yaml("../configs/data_generator.yaml")
    #
    # ## Make a log folder to log the experiments
    # make_log_folder(log_folder_name=data_generator_config["log_path"])
    #
    # ## Make a list of configs, we'll be running the model for each config
    # config_list = generate_config_list(data_generator_config)
    #
    # for i, config in enumerate(config_list):
    #     print(f"...running config {i + 1}/{len(config_list)}:")
    #     print(config)
    #     run_nn(config=config)
    #
    # # cmd = "rm -r " + "../output/nn_return/{}".format(graph_name)
    # # os.system(cmd)
