import logging
import os.path
from datetime import datetime
from testing_functions import run_routing
from pprint import pformat, pprint
import time
import numpy as np

import dgl

import data_helper
from datasets_generator import create_train_val_test_sets
from Trainer import Trainer
from utils import make_log_folder, generate_config_list
from neural_net.NeuralNet1 import NeuralNet1
import node2vec, modified_node2vec

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
from evaluations import evaluate_metrics


if __name__ == '__main__':
    ##### Here is a sample flow to run the project
    ## Firstly, change the config files in "/configs"
    ##      + data_generator.yaml: edit `file_name` to choose the input file. Pick small dataset to start first. The rest of params can be left the same
    ##      + neural_net_1.yaml: Params for neural net model
    ## Then, follow each of below steps

    ## Read the config file:
    data_generator_config = data_helper.read_yaml("../configs/routing2.yaml")

    ## Make a log folder to log the experiments
    make_log_folder(log_folder_name=data_generator_config["log_path"])

    ## Make a list of configs, we'll be running the model for each config
    config_list = generate_config_list(data_generator_config)

    ## train model with a config
    for i, config in enumerate(config_list):
        pprint(config)
        file_name = data_helper.get_file_name(config)

        t_start = time.time()
        date_time = datetime.now()
        logging.basicConfig(
            filename=os.path.join(config["log_path"], file_name + "_{}.log".format(datetime.strftime(date_time, '%Y%m%d%H%M%S_%f'))),
            level=logging.INFO)
        logging.info("Start config " + str(i + 1) + " at " + str(datetime.now()))

        # logging.info(config)

        ##### Step 1. Read data
        ## Load input file into a DGL graph, or download NetworkX graph and convert
        if config["graph"]["source"] == "osmnx":
            # OSMnx graphs have length attribute for edges, and "x" and "y" for nodes denoting longitude and latitude
            nx_graph = data_helper.download_networkx_graph(config["graph"]["name"], config["graph"]["download_type"])
            # The node attributes don't need to be kept for dgl since they won't be used in node2vec training,
            # however the edge weights are used in the modified version
            dgl_graph = dgl.from_networkx(nx_graph, edge_attrs=["length"])
        else:
            # For .gr files, distance and coordinate attributes are imported with the same names as OSMnx for consistency
            # .edgelist files are unweighted and have no spatial data, so the coordinates path with be ignored
            input_path = config["graph"]["file_path"].format(file_name=file_name)
            coord_path= config["graph"]["coord_path"].format(file_name=file_name)
            dgl_graph = data_helper.load_dgl_graph(path=input_path, undirected=True, c_path=coord_path)
            nx_graph = dgl.to_networkx(dgl_graph, edge_attrs=['length'])

    
        ##### Step 2. Run Node2Vec to get the embedding
        node2vec_args = config["node2vec"]
        print("Mean edge weight: {}".format(dgl_graph.edata['length'].mean().item()))
        print("Mean node distance: {}".format(node2vec_args["walk_length"]*dgl_graph.edata['length'].mean().item()))
        embedding_output_path = config["dataset"]["embedding_output_path"].format(name=file_name,
                                                                            epochs=node2vec_args["epochs"],
                                                                            lr=node2vec_args["lr"])
        if os.path.isfile(embedding_output_path):
            embedding = data_helper.read_file(embedding_output_path)
            print(f"Embedding already exists! Read back from {embedding_output_path}")
        else:
            embedding = node2vec.run_node2vec(dgl_graph, eval_set=None, args=node2vec_args, output_path=embedding_output_path)
        print(f"Done embedding {file_name}!")


        # ##### Step 2.5 Run Modified Node2Vec to get the embedding
        # file_name = "modified_"+file_name
        # node2vec_args = config["modified_node2vec"]
        # node2vec_args["init_c"] = node2vec_args["walk_length"]*dgl_graph.edata['length'].mean().item()
        # print("Mean node distance for modified: {}".format(node2vec_args["init_c"]))
        
        # embedding_output_path = config["dataset"]["embedding_output_path"].format(name=file_name,
        #                                                                     epochs=node2vec_args["epochs"],
        #                                                                     lr=node2vec_args["lr"])
        # print(embedding_output_path)
        # if os.path.isfile(embedding_output_path):
        #     modified_embedding = data_helper.read_file(embedding_output_path)
        #     print(f"Embedding already exists! Read back from {embedding_output_path}")
        # else:
        #     modified_embedding = modified_node2vec.run_node2vec(dgl_graph, eval_set=None, args=node2vec_args, output_path=embedding_output_path)
        # print(f"Done modified embedding {file_name}!")
        modified_embedding = None

        # set a name to save the model,  `None` (default) means not save the model
        model_name = "".join(["_file_name-", config["graph"]["name"],
                            "_sample_method-", config["dataset"]["sample_method"],
                            "_sample_ratio_for_training-", str(config["dataset"]["sample_ratio_for_training"]),
                            "_sample_ratio_for_testing-", str(config["dataset"]["sample_ratio_for_testing"])])


        ## Params for the neural net: TODO: separate model's params
        params_net1 = data_helper.read_yaml("../configs/neural_net_2.yaml")
        model=Trainer.load_model(NeuralNet1, params_net1, model_name)


        if model is None:
            ##### Step 3. Create dataset using landmarks
            train_dataset = create_train_val_test_sets(config, nx_graph, embedding, mode="train")  # for training
            test_dataset = create_train_val_test_sets(config, nx_graph, embedding, mode="test")  # for testing

            #### Step 4. Train model

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
            model, val_metrics_list, test_metrics = Trainer.train_model(NeuralNet1, train_dataset, params_net1,
                                                                            test_dataset=test_dataset,
                                                                            model_name=model_name)

            ##### Use model to approximate the distance (for demo purpose)
            emb_of_2_nodes = np.float32(np.random.rand(3, 128))
            print("Predict: ", Trainer.predict(model, emb_of_2_nodes))

            ## log metrics for val, test sets
            # logging.info("val metrics list:\n" + pformat(list(zip(range(1, len(val_metrics_list) + 1), val_metrics_list))))
            logging.info(config['dataset'])
            logging.info("loss and metrics on test set:\n" + str(test_metrics))
        else:
            train_dataset = create_train_val_test_sets(config, nx_graph, embedding, mode="train")
            val_dataset = CustomDataset(train_dataset["x_val"], train_dataset["y_val"])
            val_loader = DataLoader(dataset=val_dataset, batch_size=params_net1["batch_size"])
            loss_fn = nn.PoissonNLLLoss(log_input=False, eps=1e-07, reduction='mean')
            validation_loss, val_metrics = Trainer.evaluate_model(model, loss_fn, val_loader, 'cpu', evaluate_metrics)
            print(f"Validation loss: {validation_loss:.4f}, Validation metrics: {val_metrics}")

        ## Measure running time
        t_end = time.time()
        logging.info(f"End config {i + 1} in {round((t_end - t_start) / 60, 2)} mins, at " + str(datetime.now()))
        logging.info("\n------------\n")

        ##### Step 5. Run routing tests on 
        print("----------")
        run_routing(config, nx_graph, model, embedding, modified_embedding)