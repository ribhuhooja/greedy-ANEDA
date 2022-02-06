import os.path
from turtle import distance

import dgl
from sklearn.model_selection import train_test_split
import numpy as np
import torch

import data_helper
import landmarks
import node2vec
from data_helper import read_file
from utils import plot_nx_graph
import networkx as nx

from tqdm import tqdm


def create_train_val_test_sets(config, nx_graph, embedding, mode):
    file_name = data_helper.get_file_name(config)
    random_seed = config["random_seed"]

    if mode == "train":
        final_output_path = config["dataset"]["final_train_val_sets_path"].format(name=file_name,
                                                                               sample_ratio_for_training=
                                                                               config["dataset"][
                                                                                   "sample_ratio_for_training"],
                                                                               sample_method=config["dataset"][
                                                                                   "sample_method"],
                                                                               val_size=config["val_size"])
        force_recreate = config["force_recreate_train_and_val_sets"]

    elif mode == "test":
        final_output_path = config["dataset"]["final_test_set_path"].format(name=file_name,
                                                                         sample_ratio_for_testing=config["dataset"][
                                                                             "sample_ratio_for_testing"],
                                                                         sample_method=config["dataset"][
                                                                             "sample_method"],
                                                                         )
        force_recreate = config["force_recreate_test_set"]
    else:
        raise ValueError()

    if (not force_recreate) and os.path.isfile(final_output_path):  ## if the file already exists
        print(f"datasets for '{file_name}' already exists, only read them back!")
        return read_file(final_output_path)

    #####  Create labels
    # convert the `dgl` graph to `networkx` graph. We will use networkx for finding the shortest path

    sample_method = None

    if mode == "test":
        sample_method = 'random'
        sample_ratio = config["dataset"]["sample_ratio_for_testing"]
        random_seed = random_seed + 1  # change to a different random seed to make test set different from training sets
    elif mode == "train":
        sample_method = config["dataset"]["sample_method"]
        sample_ratio = config["dataset"]["sample_ratio_for_training"]
    else:
        raise ValueError("mode should be 'train' or 'test'!")

    if sample_method == 'random':
        num_landmarks = int(len(nx_graph) * sample_ratio)
        landmark_nodes = landmarks.get_landmark_nodes(num_landmarks, nx_graph, random_seed=random_seed)
    elif sample_method == 'high_degree':
        landmark_nodes = landmarks.get_landmark_custom(nx_graph, portion=sample_ratio)
    elif sample_method == "high_and_low_degree":
        landmark_nodes = landmarks.get_landmark_custom2(nx_graph, portion=sample_ratio)
    elif 'centrality' in sample_method:
        landmark_nodes = landmarks.get_landmark_custom3(nx_graph, portion=sample_ratio,
                                                        centrality_type=sample_method)
    elif sample_method == 'medium_degree':
        landmark_nodes = landmarks.get_landmark_custom4(nx_graph, portion=sample_ratio)

    else:
        raise ValueError(
            f"landmark sampling method should be in ['betweenness_centrality', 'closeness_centrality','random','high_degree','medium_degree'], instead of {sample_method}!")

    # TODO: when all nodes are landmark nodes, might need a better way to calc the distance (symmetric matrix)

    # Get landmarks' distance: get distance of every pair (l,n), where l is a landmark node, n is a node in the graph
    landmark_distance_output_path = config["dataset"]["landmark_distance_output_path"].format(name=file_name)
    print("Calculating landmarks distance...")
    distance_map = landmarks.calculate_landmarks_distance(config, landmark_nodes, nx_graph, 
                                                          output_path=landmark_distance_output_path)
    print("Done landmarks distance!")

    ## Plot the network
    if config["plot"]["plot_nx_graph"]:
        plot_nx_graph(nx_graph, name=file_name)

    node_to_idx = {v: i for i,v in enumerate(list(nx_graph.nodes()))}

    ##### Step 4: Create datasets to train a model
    x, y = data_helper.create_dataset(distance_map, embedding, node_to_idx)
    x, y = data_helper.remove_data_with_a_few_observations(x, y)

    print("AVERAGE LABEL: {}".format(y.mean()))

    if mode == "train":
        print("creating train and val sets for training...")

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=config["val_size"], random_state=random_seed,
                                                          shuffle=True, stratify=y)

        datasets = dict()
        datasets["x_train"] = x_train
        datasets["y_train"] = y_train
        datasets["x_val"] = x_val
        datasets["y_val"] = y_val

        if config["write_train_val_sets"]:
            print(f'writing train_val_sets datasets for {config["graph"]["name"]}...')
            data_helper.write_file(final_output_path, datasets)
            print(f'Done writing train and val sets for {config["graph"]["name"]}')

    elif mode == "test":
        print("creating a test set for testing...")
        datasets = dict()
        datasets["x_test"] = x
        datasets["y_test"] = y

        if config["write_test_set"]:
            print(f'writing test set datasets for {config["graph"]["name"]}...')
            data_helper.write_file(final_output_path, datasets)
            print(f'Done writing test set for {config["graph"]["name"]}')
    else:
        raise ValueError("mode should be 'train' or 'test'!")

    return datasets

def create_coord_dataset(config, nx_graph, node_list, node2idx):
    sources = np.random.choice(node_list, int(len(node_list) * config["coord2vec"]["sample_ratio"]), replace=False)
    targets = node_list
    
    nodes = []
    for source in tqdm(sources):
        for target in targets:
            nodes.append([node2idx[source], node2idx[target], nx_graph.nodes[source]['y'], nx_graph.nodes[target]['y'], nx_graph.nodes[source]['x'], nx_graph.nodes[target]['x']])
    
    nodes = torch.tensor(nodes)

    # Result in km
    R = 6731
    p = np.pi/180
    d = 0.5 - torch.cos((nodes[:, 3]-nodes[:, 2])*p)/2 + torch.cos(nodes[:, 2]*p)*torch.cos(nodes[:, 3]*p) * (1-torch.cos((nodes[:, 5]-nodes[:, 4])*p))/2
    distances = 2*R*torch.arcsin(torch.sqrt(d))
    
    dataset = torch.hstack((nodes[:, 0:2], distances.unsqueeze(dim=1)))
    return dataset

# def create_collab_filtering_dataset(config, nx_graph, node_list, node2idx):
#     sources = np.random.choice(node_list, int(len(node_list) * config["collab_filtering"]["sample_ratio"]), replace=False)
#     dataset = []
#     for source in tqdm(sources):
#         node_dists = nx.shortest_path_length(G=nx_graph, source=source, weight="length")
#         for node_n, dist_to_n in node_dists.items():
#             # put distance in kilometers to make training faster
#             dataset.append([node2idx[source], node2idx[node_n], dist_to_n / 1000])

#     return torch.tensor(dataset)

def create_collab_filtering_dataset(config, nx_graph, sample_ratio, node_list, node2idx):
    sources = np.random.choice(node_list, int(len(node_list) * sample_ratio), replace=False)
    node_list = set(node_list)
    dataset_map = {}
    weight = "length" if config["graph"]["source"] == "osmnx" or config["graph"]["source"] == "gis-f2e" else None
    for source in tqdm(sources):
        node_dists = nx.shortest_path_length(G=nx_graph, source=source, weight=weight)
        for node_n, dist_to_n in node_dists.items():
            if node_n in node_list: # not config["rizi_train"] or node_n in node_list # (node_n in node_list and dist_to_n > 1 and dist_to_n <= 5):
                if node_n != source and ((source, node_n) not in dataset_map or dist_to_n < dataset_map[(source, node_n)]):
                    dataset_map[(source, node_n)] = np.double(dist_to_n)
            
    dataset = []
    for (source, node_n), dist_to_n in tqdm(dataset_map.items()):
        # if config["graph"]["source"] == "osmnx" or config["graph"]["source"] == "gis-f2e":
        #     dist_to_n = dist_to_n / 1000
        dataset.append([node2idx[source], node2idx[node_n], dist_to_n])

    return torch.tensor(dataset), sources