import logging
import os.path
from datetime import datetime
from testing_functions import run_routing
from pprint import pformat, pprint

from data_helper import read_yaml
from datasets_generator import create_train_val_test_sets
from Trainer import Trainer
from utils import make_log_folder, generate_config_list
import time
from neural_net.NeuralNet1 import NeuralNet1

if __name__ == '__main__':
    ##### Here is a sample flow to run the project
    ## Firstly, change the config files in "/configs"
    ##      + data_generator.yaml: edit `file_name` to choose the input file. Pick small dataset to start first. The rest of params can be left the same
    ##      + neural_net_1.yaml: Params for neural net model
    ## Then, follow each of below steps

    ## Read the config file:
    data_generator_config = read_yaml("../configs/data_generator.yaml")

    ## Make a log folder to log the experiments
    make_log_folder(log_folder_name=data_generator_config["log_path"])

    ## Make a list of configs, we'll be running the model for each config
    config_list = generate_config_list(data_generator_config)

    ## train model with a config
    for i, config in enumerate(config_list):
        pprint(config)

        t_start = time.time()
        date_time = datetime.now()
        logging.basicConfig(
            filename=os.path.join(config["log_path"], config["data"]["file_name"] + "_{}.log".format(datetime.strftime(date_time, '%Y%m%d%H%M%S_%f'))),
            level=logging.INFO)
        logging.info("Start config " + str(i + 1) + " at " + str(datetime.now()))

        # logging.info(config)

        train_dataset = create_train_val_test_sets(config, mode="train")  # for training
        test_dataset = create_train_val_test_sets(config, mode="test")  # for testing

        ## Params for the neural net: TODO: separate model's params
        params_net1 = read_yaml("../configs/neural_net_1.yaml")

        #### Train model
        # set a name to save the model,  `None` (default) means not save the model
        model_name = "".join(["_file_name-", config["data"]["file_name"],
                              "_sample_method-", config["landmark"]["sample_method"],
                              "_sample_ratio_for_training-", str(config["landmark"]["sample_ratio_for_training"]),
                              "_sample_ratio_for_testing-", str(config["landmark"]["sample_ratio_for_testing"])])

        ## Train
        # model, val_metrics_list, test_metrics = Trainer.train_model(NeuralNet1, train_dataset, params_net1,
        #                                                             test_dataset=test_dataset, model_name=model_name)
        #
        # ## If we want to load a saved model:
        # new_model = Trainer.load_model(NeuralNet1, params=params_net1, model_name=model_name)
        #
        # ## Sanity check
        # assert Trainer.compare_2_models(model, new_model), "The 2 models are different!"

        ## We can also call Trainer.maybe_train_model:
        model, val_metrics_list, test_metrics = Trainer.maybe_train_model(NeuralNet1, train_dataset, params_net1,
                                                                          test_dataset=test_dataset,
                                                                          model_name=model_name)

        ## log metrics for val, test sets
        # logging.info("val metrics list:\n" + pformat(list(zip(range(1, len(val_metrics_list) + 1), val_metrics_list))))
        logging.info(config['landmark'])
        logging.info("loss and metrics on test set:\n" + str(test_metrics))

        ## Measure running time
        t_end = time.time()
        logging.info(f"End config {i + 1} in {round((t_end - t_start) / 60, 2)} mins, at " + str(datetime.now()))
        logging.info("\n------------\n")

        ##########
        ### Test to see if the routing code can run
        print("----------")
        run_routing(config=config)
