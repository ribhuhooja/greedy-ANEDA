import logging
import os.path
from datetime import datetime
import testing_functions
from pprint import pformat, pprint
import time
import numpy as np

import dgl

import data_helper
from datasets_generator import create_train_val_test_sets, create_collab_filtering_dataset
from Trainer import Trainer
from utils import make_log_folder, generate_config_list
from neural_net.NeuralNet1 import NeuralNet1
import collaborative_filtering

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
    data_generator_config = data_helper.read_yaml("../configs/collab_filtering.yaml")

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
        
        node_list = list(nx_graph.nodes)
        node2idx = {v:i for i,v in enumerate(node_list)}

        #### Step 2. Get initial vectors from coordinates
        coord_embedding = data_helper.get_coord_embedding(nx_graph, node_list)

        ##### Step 2. Run Collaborative Filtering using initial embedding to get final embeddings
        collab_filtering_args = config["collab_filtering"]
        assert collab_filtering_args["embedding_dim"] >= coord_embedding.shape[1]

        embedding_output_path = "../output/embedding/{name}_embed-epochs{epochs}-lr{lr}-ratio{ratio}-d{dim}{hyperbolic}.pkl".format(name="collab_filtering/"+file_name,
                                                                            epochs=collab_filtering_args["epochs"],
                                                                            lr=collab_filtering_args["lr"],
                                                                            ratio=collab_filtering_args["sample_ratio"],
                                                                            dim=collab_filtering_args["embedding_dim"],
                                                                            hyperbolic="-h" if collab_filtering_args["hyperbolic"] else "")
        if os.path.isfile(embedding_output_path):
            embedding = data_helper.read_file(embedding_output_path)
            print(f"Embedding already exists! Read back from {embedding_output_path}")
        else:
            print("Creating dataset")
            dataset_output_path = "../output/datasets/collab_filtering_{}_ratio-{}".format(file_name, collab_filtering_args["sample_ratio"])

            if os.path.isfile(dataset_output_path):
                collab_filtering_dataset = data_helper.read_file(dataset_output_path)
                print(f"Dataset already exists! Read back from {dataset_output_path}")
            else:
                collab_filtering_dataset = create_collab_filtering_dataset(nx_graph, collab_filtering_args["sample_ratio"], node_list, node2idx)
                data_helper.write_file(dataset_output_path, collab_filtering_dataset)
            print("Finished dataset")
            # /collab_filtering_args["embedding_dim"]
            init_embedding = np.random.normal(scale=1/collab_filtering_args["embedding_dim"], size=(len(nx_graph.nodes), collab_filtering_args["embedding_dim"]))
            if collab_filtering_args["hyperbolic"]:
                coord_embedding = np.divide(coord_embedding, np.linalg.norm(coord_embedding, axis=1)[:, None]) * (np.sqrt(2)-1)
            init_embedding[:, 0:coord_embedding.shape[1]] = coord_embedding
            # if collab_filtering_args["hyperbolic"]:
            #     init_embedding = np.divide(init_embedding, np.linalg.norm(init_embedding, axis=1)[:, None]) * (np.sqrt(2)-1)
            # print(np.mean(init_embedding, axis=0), np.var(init_embedding, axis=0))
            
            init_embedding = torch.from_numpy(init_embedding)

            embedding = collaborative_filtering.run_collab_filtering(collab_filtering_dataset, len(nx_graph.nodes), init_embeddings=init_embedding, eval_set=None, args=collab_filtering_args, output_path=embedding_output_path)
        print(f"Done embedding {file_name}!")

        # Generate all route pairs for Belmont CA to output complete performance percentiles
        testing_functions.run_routing_embedding(config, nx_graph, embedding, test_pairs=True, plot_route=False, run_dijkstra=False, run_dist=False, pairs_to_csv=True)

        # Plot routing for specific source and target, and compare to distance
        # source, target = 65521256, 6728059433
        # testing_functions.run_routing_embedding(config, nx_graph, embedding, test_pairs=False, plot_route=True, run_dijkstra=False, run_dist=True, source=source, target=target)
