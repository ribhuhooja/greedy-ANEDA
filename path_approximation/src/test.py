from cgi import test
from distutils.file_util import write_file
import logging
import os.path
from datetime import datetime
import testing_functions
from pprint import pformat, pprint
import time
import numpy as np

import dgl
import networkx as nx

import data_helper
from datasets_generator import create_train_val_test_sets, create_collab_filtering_dataset
from Trainer import Trainer
from utils import make_log_folder, generate_config_list
from neural_net.NeuralNet1 import NeuralNet1
import collaborative_filtering
import node2vec
from grarep import GraRep

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
        np.random.seed(config["random_seed"])

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
        elif config["graph"]["source"] == "gis-f2e":
            input_path = "../data/{name}.csv".format(name=config["graph"]["name"])
            nx_graph = data_helper.load_gis_f2e_to_nx_graph(input_path)
            dgl_graph = dgl.from_networkx(nx_graph, edge_attrs=["length"])   
        else:
            # For .gr files, distance and coordinate attributes are imported with the same names as OSMnx for consistency
            # .edgelist files are unweighted and have no spatial data, so the coordinates path with be ignored
            input_path = config["graph"]["file_path"].format(name=file_name, source=config["graph"]["source"])
            coord_path= config["graph"]["coord_path"].format(name=file_name)
            dgl_graph = data_helper.load_dgl_graph(path=input_path, undirected=True, c_path=coord_path)
            nx_graph = dgl.to_networkx(dgl_graph, edge_attrs=['length'])
        
        node_list = np.asarray(nx_graph.nodes)
        node2idx = {v:i for i,v in enumerate(node_list)}

        ##### Run Collaborative Filtering using initial embedding to get final embeddings
        collab_filtering_args = config["collab_filtering"]

        embedding_output_path = "../output/embedding/collab_filtering/{name}_embed-epochs{epochs}-lr{lr}-ratio{ratio}-d{dim}{measure}{norm}{embedding}.pkl".format(name=file_name,
                                                                            epochs=collab_filtering_args["epochs"],
                                                                            lr=collab_filtering_args["lr"],
                                                                            ratio=collab_filtering_args["sample_ratio"],
                                                                            dim=collab_filtering_args["embedding_dim"],
                                                                            measure="-"+collab_filtering_args["measure"] if collab_filtering_args["measure"] != "norm" else "",
                                                                            norm="-p"+str(collab_filtering_args["norm"]) if collab_filtering_args["measure"] == "norm" and collab_filtering_args["norm"] != 2 else "",
                                                                            embedding=("-"+collab_filtering_args["init_embedding"]+(str(config["grarep"]["order"]) if collab_filtering_args["init_embedding"] == "grarep" else "")) if collab_filtering_args["init_embedding"] != "coord" else "")
        #### Step 2. Create datasets
        print("Creating dataset")
        dataset_output_path = "../output/datasets/collab_filtering_{}_ratio-{}".format(file_name, collab_filtering_args["sample_ratio"])
        test_dataset_output_path = "../output/datasets/collab_filtering_test_{}_ratio-{}".format(file_name, collab_filtering_args["test_sample_ratio"])

        if os.path.isfile(dataset_output_path) and os.path.isfile(test_dataset_output_path):
            collab_filtering_dataset = data_helper.read_file(dataset_output_path)
            test_dataset = data_helper.read_file(test_dataset_output_path)

            collab_filtering_dataset = collab_filtering_dataset[collab_filtering_dataset[:, 0] != collab_filtering_dataset[:, 1]]
            test_dataset = test_dataset[test_dataset[:, 0] != test_dataset[:, 1]]
            print(f"Dataset already exists! Read back from {dataset_output_path}")
        else:
            assert collab_filtering_args["sample_ratio"] + collab_filtering_args["test_sample_ratio"] <= 1
            collab_filtering_dataset, sources = create_collab_filtering_dataset(config, nx_graph, collab_filtering_args["sample_ratio"], node_list, node2idx)
            rem_node_list = np.setdiff1d(node_list, sources)
            if collab_filtering_args["sample_ratio"] == 1:
                test_dataset = torch.clone(collab_filtering_dataset)
            else:
                test_dataset, _ = create_collab_filtering_dataset(config, nx_graph, collab_filtering_args["test_sample_ratio"]/(1-collab_filtering_args["sample_ratio"]), rem_node_list, node2idx)
            data_helper.write_file(dataset_output_path, collab_filtering_dataset)
            data_helper.write_file(test_dataset_output_path, test_dataset)
        print("Finished dataset. Size:", len(collab_filtering_dataset), ",", len(test_dataset))

        
        if os.path.isfile(embedding_output_path):
            embedding = data_helper.read_file(embedding_output_path)
            print(f"Embedding already exists! Read back from {embedding_output_path}")
        else:
            #### Step 3. Get initial vectors from graph
            full_init_embedding = np.random.normal(scale=1/collab_filtering_args["embedding_dim"], size=(len(nx_graph.nodes), collab_filtering_args["embedding_dim"]))
            h = 1.1
            assert config["collab_filtering"]["init_embedding"] in ["none", "coord", "node2vec", "grarep"]
            if config["collab_filtering"]["init_embedding"] != "none":
                if config["collab_filtering"]["init_embedding"] == "coord":
                    assert config["graph"]["source"] == "osmnx" or config["graph"]["source"] == "gis-f2e"
                    init_embedding = data_helper.get_coord_embedding(config, nx_graph, node_list)
                    assert collab_filtering_args["embedding_dim"] >= init_embedding.shape[1]
                elif config["collab_filtering"]["init_embedding"] == "node2vec":
                    ##### Run Node2Vec to get the embedding if not spatial (or no coordinates)
                    node2vec_args = config["node2vec"]
                    assert collab_filtering_args["embedding_dim"] >= node2vec_args["embedding_dim"]
                    node2vec_output_path = "../output/embedding/node2vec/{name}_embed-epochs{epochs}-lr{lr}-d{dim}.pkl".format(name=file_name,
                                                                                epochs=node2vec_args["epochs"],
                                                                                lr=node2vec_args["lr"],
                                                                                dim=node2vec_args["embedding_dim"])
                    if os.path.isfile(node2vec_output_path):
                        init_embedding = data_helper.read_file(node2vec_output_path)
                        print(f"Embedding already exists! Read back from {node2vec_output_path}")
                    else:
                        init_embedding = node2vec.run_node2vec(dgl_graph, eval_set=None, args=node2vec_args, output_path=node2vec_output_path)
                    print(f"Done embedding {file_name}!")

                    # Removing padding idx that is no longer necessary
                    init_embedding = init_embedding[1:]
                elif config["collab_filtering"]["init_embedding"] == "grarep":
                    grarep_args = config["grarep"]
                    assert collab_filtering_args["embedding_dim"] >= grarep_args["embedding_dim"]
                    grarep_output_path = "../output/embedding/grarep/{name}_embed-order{order}-iters{iters}-d{dim}.pkl".format(name=file_name,
                                                                                order=grarep_args["order"],
                                                                                iters=grarep_args["iterations"],
                                                                                dim=grarep_args["embedding_dim"])
                    if os.path.isfile(grarep_output_path):
                        init_embedding = data_helper.read_file(grarep_output_path)
                        print(f"Embedding already exists! Read back from {grarep_output_path}")
                    else:
                        weight = "length" if config["graph"]["source"] == "osmnx" else "weight"
                        grarep = GraRep(grarep_args["embedding_dim"], grarep_args["iterations"], grarep_args["order"], weight=weight)
                        G = nx.relabel_nodes(nx_graph, node2idx)
                        grarep.fit(G)
                        init_embedding = grarep.get_embedding()
                        data_helper.write_file(grarep_output_path, init_embedding)
                    print(f"Done embedding {file_name}!")

                #### Step 3a. Transform and use inital embedding, padding to desired dimension with random normals
                if collab_filtering_args["measure"] == "hyperbolic":
                    init_embedding = np.divide(init_embedding, np.linalg.norm(init_embedding, axis=1)[:, None]) * h # * (np.sqrt(2)-1)
                elif collab_filtering_args["measure"] == "spherical":
                    init_embedding = np.divide(init_embedding, np.linalg.norm(init_embedding, axis=1)[:, None])
                full_init_embedding[:, 0:init_embedding.shape[1]] = init_embedding    
            elif collab_filtering_args["measure"] == "hyperbolic":
                full_init_embedding = np.divide(full_init_embedding, np.linalg.norm(full_init_embedding, axis=1)[:, None]) * h

            full_init_embedding = torch.from_numpy(full_init_embedding)

            #### Step 4. Run collaborative filtering
            embedding = collaborative_filtering.run_collab_filtering(collab_filtering_dataset, len(nx_graph.nodes), init_embeddings=full_init_embedding, eval_set=test_dataset, args=collab_filtering_args, output_path=embedding_output_path, config=config)
        print(f"Done embedding {file_name}!")

        test_metrics = collaborative_filtering.test_collab_filtering(config, embedding, test_dataset)
        print("Final Results: {}".format(test_metrics))

        #### Step 5. Run routing
        # Generate all route pairs for Belmont CA to output complete performance percentiles
        if config["run_routing"]:
            testing_functions.run_routing_embedding(config, nx_graph, embedding, test_pairs=True, plot_route=False, run_dijkstra=False, run_dist=False, pairs_to_csv=True)

        # Test alphas, reporting stretch
        # testing_functions.run_routing_embedding(config, nx_graph, embedding, test_pairs=True, plot_route=False, run_dijkstra=False, run_dist=False, pairs_to_csv=True, alpha=1.75, report_stretch=True)

        # Plot routing for specific source and target, and compare to distance
        # source, target = 178318511, 1075324802
        # testing_functions.run_routing_embedding(config, nx_graph, embedding, test_pairs=False, plot_route=True, run_dijkstra=False, run_dist=True, source=source, target=target)
