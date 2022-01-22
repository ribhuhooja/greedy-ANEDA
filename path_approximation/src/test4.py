import logging
import os.path
from datetime import datetime
import testing_functions
from pprint import pprint
import time
import numpy as np
from tqdm import tqdm

import dgl
import data_helper
from datasets_generator import create_collab_filtering_dataset
from utils import make_log_folder, generate_config_list


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

        print("Creating matrix")
        matrix_output_path = "../output/matrix/{}_distance_matrix".format(file_name)
        if os.path.isfile(matrix_output_path):
            matrix = data_helper.read_file(matrix_output_path)
            print(f"Matrix already exists! Read back from {matrix_output_path}")
        else:                       
            matrix = data_helper.get_dist_matrix(nx_graph, node_list, node2idx)
            data_helper.write_file(matrix_output_path, matrix)
        print("Finished matrix")
        
        # Generate all route pairs for Belmont CA to output complete performance percentiles
        testing_functions.run_routing_dist_matrix(config, nx_graph, matrix, test_pairs=True, plot_route=False, run_dijkstra=False, run_dist=False, pairs_to_csv=True)

        # Plot routing for specific source and target, and compare to distance
        # source, target = 178362470, 178385772
        # testing_functions.run_routing_dist_matrix(config, nx_graph, matrix, test_pairs=False, plot_route=True, run_dijkstra=False, run_dist=False, source=source, target=target)
