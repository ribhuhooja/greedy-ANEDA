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
from datasets_generator import create_aneda_dataset
from utils import make_log_folder, generate_config_list
import aneda
import node2vec
from grarep import GraRep

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from evaluations import evaluate_metrics



def main(config_path):
     ##### Here is a sample flow to run the project
    ## Firstly, change the config files in "/configs"
    ##      + data_generator.yaml: edit `file_name` to choose the input file. Pick small dataset to start first. The rest of params can be left the same
    ##      + neural_net_1.yaml: Params for neural net model
    ## Then, follow each of below steps

    ## Read the config file:
    data_generator_config = data_helper.read_yaml(config_path)

    ## Make a log folder to log the experiments
    make_log_folder(log_folder_name=data_generator_config["log_path"])

    config = generate_config_list(data_generator_config)[0]

    #pprint(config)
    file_name = data_helper.get_file_name(config)
    rng = np.random.default_rng(config["random_seed"])

    t_start = time.time()
    date_time = datetime.now()
    logging.basicConfig(
        filename=os.path.join(config["log_path"], file_name + "_{}.log".format(datetime.strftime(date_time, '%Y%m%d%H%M%S_%f'))),
        level=logging.INFO)
    logging.info("Start config at " + str(datetime.now()))

    # logging.info(config)

    ##### Step 1. Read data
    ## Load input file into a DGL graph, or download NetworkX graph and convert
    # OSMnx graphs have length attribute for edges, and "x" and "y" for nodes denoting longitude and latitude
    if config["graph"]["source"] == "osmnx":
        if 'bbox' in config['graph']:
            nx_graph = data_helper.download_networkx_graph_bbox(
                config["graph"]["name"], config["graph"]["download_type"], 
                config["graph"]["bbox"]["north"], config["graph"]["bbox"]["south"], config["graph"]["bbox"]["east"], config["graph"]["bbox"]["west"])
        else:
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

    if config["run_dist_routing"]:
        testing_functions.run_routing_dist(config, nx_graph, test_pairs=True, plot_route=False, pairs_to_csv=True) #changed plot to false for now
        #exit()

    ##### Run ANEDA using initial embedding to get final embeddings
    aneda_args = config["aneda"]

    embedding_output_path = "../output/embedding/aneda/{name}/{loss}epochs{epochs}-lr{lr}-ratio{ratio}-d{dim}{measure}{norm}{embedding}.pkl".format(name=file_name,
                                                                        loss=aneda_args["loss_func"]+"_" if aneda_args["loss_func"] != "mre" and aneda_args["loss_func"] != "mse" else "",
                                                                        epochs=aneda_args["epochs"],
                                                                        lr=aneda_args["lr"],
                                                                        ratio=aneda_args["sample_ratio"],
                                                                        dim=aneda_args["embedding_dim"],
                                                                        measure="-"+aneda_args["measure"] if aneda_args["measure"] != "norm" else "",
                                                                        norm="-p"+str(aneda_args["norm"]) if aneda_args["measure"] == "norm" and aneda_args["norm"] != 2 else "",
                                                                        embedding=("-"+aneda_args["init_embedding"]+(str(config["grarep"]["order"]) if aneda_args["init_embedding"] == "grarep" else "")) if aneda_args["init_embedding"] != "coord" else "")
    print("Full embedding output path:", embedding_output_path)
    #### Step 2. Create datasets
    print("Creating dataset")
    dataset_output_path = "../output/datasets/aneda/{}/ratio-{}".format(file_name, aneda_args["sample_ratio"])
    test_dataset_output_path = "../output/datasets/aneda/{}/test_ratio-{}".format(file_name, aneda_args["test_sample_ratio"])

    if os.path.isfile(dataset_output_path) and os.path.isfile(test_dataset_output_path):
        aneda_dataset = data_helper.read_file(dataset_output_path)
        test_dataset = data_helper.read_file(test_dataset_output_path)

        aneda_dataset = aneda_dataset[aneda_dataset[:, 0] != aneda_dataset[:, 1]]
        test_dataset = test_dataset[test_dataset[:, 0] != test_dataset[:, 1]]
        print(f"Dataset already exists! Read back from {dataset_output_path}")
    else:
        assert aneda_args["sample_ratio"] + aneda_args["test_sample_ratio"] <= 1
        aneda_dataset, sources = create_aneda_dataset(config, nx_graph, aneda_args["sample_ratio"], rng, node_list, node2idx)
        rem_node_list = np.setdiff1d(node_list, sources)
        if aneda_args["sample_ratio"] == 1:
            test_dataset = torch.clone(aneda_dataset)
        else:
            test_dataset, _ = create_aneda_dataset(config, nx_graph, aneda_args["test_sample_ratio"]/(1-aneda_args["sample_ratio"]), rng, rem_node_list, node2idx)
        data_helper.write_file(dataset_output_path, aneda_dataset)
        data_helper.write_file(test_dataset_output_path, test_dataset)
    print("Finished dataset. Size:", len(aneda_dataset), ",", len(test_dataset))

    
    if os.path.isfile(embedding_output_path):
        embedding = data_helper.read_file(embedding_output_path)
        print(f"Embedding already exists! Read back from {embedding_output_path}")
    else:
        #### Step 3. Get initial vectors from graph
        assert config["aneda"]["init_embedding"] in ["none", "coord", "node2vec", "grarep"]

        full_init_embedding = rng.normal(scale=1/aneda_args["embedding_dim"], size=(len(nx_graph.nodes), aneda_args["embedding_dim"]))
        init_embedding = None
        if config["aneda"]["init_embedding"] == "coord":
            assert config["graph"]["source"] == "osmnx" or config["graph"]["source"] == "gis-f2e"
            init_embedding = data_helper.get_coord_embedding(config, nx_graph, node_list)
            assert aneda_args["embedding_dim"] >= init_embedding.shape[1]
        elif config["aneda"]["init_embedding"] == "node2vec":
            ##### Run Node2Vec to get the embedding if not spatial (or no coordinates)
            node2vec_args = config["node2vec"]
            assert aneda_args["embedding_dim"] >= node2vec_args["embedding_dim"]
            node2vec_output_path = "../output/embedding/node2vec/{name}/epochs{epochs}-lr{lr}-d{dim}.pkl".format(name=file_name,
                                                                        epochs=node2vec_args["epochs"],
                                                                        lr=node2vec_args["lr"],
                                                                        dim=node2vec_args["embedding_dim"])
            if os.path.isfile(node2vec_output_path):
                init_embedding = data_helper.read_file(node2vec_output_path)
                print(f"Embedding already exists! Read back from {node2vec_output_path}")
            else:
                init_embedding = node2vec.run_node2vec(dgl_graph, eval_set=None, args=node2vec_args, output_path=node2vec_output_path, device=config["aneda"]["device"])
            print(f"Done embedding {file_name}!")

            # Removing padding idx that is no longer necessary
            init_embedding = init_embedding[1:]
        elif config["aneda"]["init_embedding"] == "grarep":
            grarep_args = config["grarep"]
            assert aneda_args["embedding_dim"] >= grarep_args["embedding_dim"]
            grarep_output_path = "../output/embedding/grarep/{name}/order{order}-iters{iters}-d{dim}.pkl".format(name=file_name,
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

        if init_embedding is not None:
            full_init_embedding[:, 0:init_embedding.shape[1]] = init_embedding
        if aneda_args["measure"] == "poincare":
            full_init_embedding = np.divide(full_init_embedding, np.linalg.norm(full_init_embedding, axis=1)[:, None]) * (np.sqrt(2)-1)  
        if aneda_args["measure"] == "hyperboloid":
            full_init_embedding = np.divide(full_init_embedding, np.linalg.norm(full_init_embedding, axis=1)[:, None]) * 1.1  

        full_init_embedding = torch.from_numpy(full_init_embedding)

        #### Step 4. Run ANEDA
        embedding = aneda.run_aneda(aneda_dataset, len(nx_graph.nodes), init_embeddings=full_init_embedding, eval_set=test_dataset, args=aneda_args, output_path=embedding_output_path, config=config)
    print(f"Done embedding {file_name}!")

    return config, nx_graph, embedding

if __name__ == '__main__':

    # Steps 1-4
    config, nx_graph, embedding = main("../configs/greedy.yaml")
    
    #### Step 5. Run routing
    # Generate all route pairs for Belmont CA to output complete performance percentiles
    if config["run_routing"]:
        testing_functions.run_routing_embedding(config, nx_graph, embedding, test_pairs=True, plot_route=False, run_dijkstra=False, run_dist=False, pairs_to_csv=True, report_stretch=True)
    if config["plot_routes"]:
        testing_functions.run_routing_embedding(config, nx_graph, embedding, test_pairs=False, plot_route=True, run_dijkstra=False, run_dist=False)
    if config["run_time_test"]:
        testing_functions.run_time_test(config, nx_graph, embedding)
    if config["run_dist_time_test"]:
        testing_functions.run_time_test(config, nx_graph, embedding, use_dist=True)

    if config["run_greedy"]:
        # choose the algorithm for routing
        greedy_algorithm = config["greedy_algorithm"]
        allowed_greedy_algos = set(["normal", "early_abort", "panic_jump"]) # This is probably the wrong place to do this but oh well
        if greedy_algorithm not in allowed_greedy_algos: 
            print("Unidentifiable greedy algorithm detected, reverting to normal greedy search")
            greedy_algorithm = "normal"


        print()
        print("running greedy from main")
        ratio_pairs = 1 if not config["greedy_pairs_ratio"] else float(config["greedy_pairs_ratio"])
        testing_functions.run_greedy(config, nx_graph, embedding, alpha=1.5, ratio_pairs=ratio_pairs, greedy_algorithm=greedy_algorithm)

    if config["evaluate_embedding_greediness"]:
        print()
        print("Evaluate greediness of embedding")
        ratio_pairs = 1 if not config["greedy_pairs_ratio"] else float(config["greedy_pairs_ratio"])
        ratio_nodes = 1 if not config["greedy_nodes_ratio"] else float(config["greedy_nodes_ratio"])
        testing_functions.evaluate_embedding_greediness(config, nx_graph, embedding, ratio_pairs, ratio_nodes)

    # Test alphas, reporting stretch
    # testing_functions.run_routing_embedding(config, nx_graph, embedding, test_pairs=True, plot_route=False, run_dijkstra=False, run_dist=False, pairs_to_csv=True, alpha=1.75, report_stretch=True)

    # Plot routing for specific source and target, and compare to distance
    # source, target = 178318511, 1075324802
    # testing_functions.run_routing_embedding(config, nx_graph, embedding, test_pairs=False, plot_route=True, run_dijkstra=False, run_dist=True, source=source, target=target)
