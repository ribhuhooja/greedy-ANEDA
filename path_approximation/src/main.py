import logging
import os.path
from datetime import datetime

from data_helper import read_yaml
from datasets_generator import create_train_val_test_sets
from train_neural_net import train_neural_net
from utils import make_log_folder, generate_config_list

if __name__ == '__main__':
    ## Read the config file:
    data_generator_config = read_yaml("../configs/data_generator.yaml")

    ## Make a log folder to log the experiments
    make_log_folder(log_folder_name=data_generator_config["log_path"])

    # Make a list of configs, we'll be running the model for each config
    config_list = generate_config_list(data_generator_config)

    dataset = create_train_val_test_sets(data_generator_config)

    for i, config in enumerate(config_list):
        logging.basicConfig(filename=os.path.join(config["log_path"], "running_log.log"), level=logging.INFO)
        logging.info("Start: " + config["data"]["file_name"] + " at " + datetime.now().strftime("%m/%d/%Y %H:%M:%S "))

        dataset = create_train_val_test_sets(config)
        val_metrics = train_neural_net(dataset)
        logging.info(val_metrics)
        logging.info("End: " + datetime.now().strftime("%m/%d/%Y %H:%M:%S "))
        logging.info("\n------------\n")
