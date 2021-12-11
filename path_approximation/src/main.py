import logging
import os.path
from datetime import datetime
from testing_functions import run_some_linear_models, run_nn
from pprint import pformat, pprint

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
    # resolved! Fix a small bug. Can now use the `config_list`.
    
    ## here just for the program to run 
    config = config_list[0]
    real_train = [0.2,0.3,0.4,0.5,0.6,0.7]
    real_method = ['betweenness_centrality', 'closeness_centrality','random','high_degree','medium_degree']
    for i in (range(len(real_train) * len(real_method))):
        #config['landmark']['sample_ratio_for_training']
        config['landmark']['sample_ratio_for_training'] = real_train[i%len(real_train)]
        config['landmark']['sample_method'] = real_method[i//len(real_train)]
        t_start = time.time()
        date_time = datetime.now()
        logging.basicConfig(
            filename=os.path.join(config["log_path"], config["data"]["file_name"] + "_" + str(date_time) + ".log"),
            level=logging.INFO)
        logging.info("Start config " + str(i + 1) + " at " + str(datetime.now()))

        #logging.info(config)

        dataset = create_train_val_test_sets(config, mode="train")  # for training
        test_dataset = create_train_val_test_sets(config, mode="test")  # for testing

        ## Params for the neural net: TODO: separate model's params
        params_net1 = read_yaml("../configs/neural_net_1.yaml")

        ## Train model
        val_metrics_list, test_metrics = Trainer.train_model(NeuralNet1, dataset, params_net1, test_dataset)

        ## write log for val_metrics
        
        #logging.info("val metrics list:\n" + pformat(list(zip(range(1, len(val_metrics_list) + 1), val_metrics_list))))
        logging.info(config['landmark'])
        logging.info("loss and metrics on test set:\n" + str(test_metrics))

        ## Measure running time
        t_end = time.time()
        logging.info(f"End config {i + 1} in {round((t_end - t_start) / 60, 2)} mins, at " + str(datetime.now()))
        logging.info("\n------------\n")
