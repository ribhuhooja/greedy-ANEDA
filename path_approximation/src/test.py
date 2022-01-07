import logging
import os.path
from datetime import datetime
from testing_functions import run_routing
from pprint import pformat, pprint
import time
import numpy as np

import dgl

import data_helper
from datasets_generator import create_train_val_test_sets, create_coord_dataset, create_collab_filtering_dataset
from Trainer import Trainer
from utils import make_log_folder, generate_config_list
from neural_net.NeuralNet1 import NeuralNet1
import coord2vec

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

    
        ##### Step 2. Run Coord2Vec to get the embedding
        coord2vec_args = config["coord2vec"]
        embedding_output_path = "../output/embedding/{name}_embed-epochs{epochs}-lr{lr}-ratio{ratio}-d.pkl".format(name="coord2vec_"+file_name,
                                                                            epochs=coord2vec_args["epochs"],
                                                                            lr=coord2vec_args["lr"],
                                                                            ratio=coord2vec_args["sample_ratio"])
        if os.path.isfile(embedding_output_path):
            coord_embedding = data_helper.read_file(embedding_output_path)
            print(f"Embedding already exists! Read back from {embedding_output_path}")
        else:
            print("Creating dataset")
            dataset_output_path = "../output/datasets/coord2vec_{}_ratio-{}".format(file_name, config["coord2vec"]["sample_ratio"])

            if os.path.isfile(dataset_output_path):
                coord_dataset = data_helper.read_file(dataset_output_path)
                print(f"Dataset already exists! Read back from {dataset_output_path}")
            else:
                coord_dataset = create_coord_dataset(config, nx_graph, node_list, node2idx)
                data_helper.write_file(dataset_output_path, coord_dataset)
            print("Finished dataset")
            print("LENGTH:", coord_dataset.size(0))

            coord_embedding = coord2vec.run_coord2vec(coord_dataset, len(nx_graph.nodes), eval_set=None, args=coord2vec_args, output_path=embedding_output_path)
        print(f"Done embedding {file_name}!")

        # loss_fn = nn.PoissonNLLLoss(log_input=False, eps=1e-07, reduction='mean')
        loss_fn = nn.MSELoss(reduction='mean')
        def real_distance(a, b):
            R = 6731
            p = np.pi/180
            lat_a, long_a, lat_b, long_b = nx_graph.nodes[a]['y'], nx_graph.nodes[a]['x'], nx_graph.nodes[b]['y'], nx_graph.nodes[b]['x'],
            
            d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
            D = 2*R*np.arcsin(np.sqrt(d))
            return torch.tensor(D)
        def coord_emb_distance(a, b):
            x, y = node2idx[a], node2idx[b]
            left, right = coord_embedding[x], coord_embedding[y]
            out = np.exp(-np.dot(left, right))
            return torch.tensor(out)

        errors = []
        for _ in range(1000):
            node1, node2 = np.random.choice(node_list), np.random.choice(node_list)
            errors.append(loss_fn(coord_emb_distance(node1, node2), real_distance(node1, node2)))
        error = "Coord MSE: {}".format((sum(errors)/len(errors)).item())
        with open("../output/log.txt", 'w') as f:
            f.write(error)
        print(error)

        ##### Step 3. Run Collaborative Filtering using initial embedding to get final embeddings
        collab_filtering_args = config["collab_filtering"]
        assert collab_filtering_args["embedding_dim"] >= coord2vec_args["embedding_dim"]

        embedding_output_path = "../output/embedding/{name}_embed-epochs{epochs}-lr{lr}-ratio{ratio}-d.pkl".format(name="collab_filtering_"+file_name,
                                                                            epochs=collab_filtering_args["epochs"],
                                                                            lr=collab_filtering_args["lr"],
                                                                            ratio=collab_filtering_args["sample_ratio"])
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
                collab_filtering_dataset = create_collab_filtering_dataset(config, nx_graph, node_list, node2idx)
                data_helper.write_file(dataset_output_path, collab_filtering_dataset)
            print("Finished dataset")

            init_embedding = np.random.normal(size=(len(nx_graph.nodes), collab_filtering_args["embedding_dim"]))
            init_embedding[:, 0:coord2vec_args["embedding_dim"]] = coord_embedding
            init_embedding = torch.from_numpy(init_embedding)

            embedding = coord2vec.run_coord2vec(collab_filtering_dataset, len(nx_graph.nodes), init_embeddings=init_embedding, eval_set=None, args=collab_filtering_args, output_path=embedding_output_path)
        print(f"Done embedding {file_name}!")
