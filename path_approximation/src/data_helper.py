import json
import os
from collections import Counter
from typing import Dict
import yaml

import dgl
import networkx as nx
import osmnx as ox

import numpy as np
import scipy
import dill
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm



def load_edgelist_file_to_dgl_graph(path: str, undirected: bool, edge_weights=None):
    """
    Reads an edgeList file in which each row contains an edge of the network, then returns a DGL graph.
    :param path: path to the edgeList file
    edgeList file  should contain 2 columns as follows:
        0 276
        0 58
        0 132

    :param path:
    :param undirected:
    :param edge_weights:

    :return: a DGL graph
    """
    input_np = np.loadtxt(path, dtype=np.int)

    # if the edgeList file starts from some number rather than 0, we will subtract that number from the indices
    min_index = np.min(input_np)
    input_np = input_np - min_index  ## make all indices start from 0
    row_indices, col_indices = input_np[:, 0], input_np[:, 1]

    if edge_weights is None:
        edge_weights = np.ones(input_np.shape[0])  # setting all the weights to 1(s)
    dim = np.max(input_np) + 1

    input_mx = scipy.sparse.coo_matrix((edge_weights, (row_indices, col_indices)), shape=(dim, dim))
    g = dgl.from_scipy(input_mx)

    if undirected:  # convert directed graph (as default, all the edges are directed in DGL) to undirected graph
        g = dgl.to_bidirected(g)

    g.edata['length'] = torch.ones(g.num_edges(),)
    return g

def load_gr_file_to_dgl_graph(path: str, undirected: bool, c_path=""):
    """
    Reads a .gr file in which each row contains an edge of the network, then returns a DGL graph.
    :param path: path to the edgeList file
    gr file  should contain 4 columns as follows:
        a 0 276 803
        a 0 58 774
        a 0 132 1400

    :param path:
    :param d_path: graph containing the true distances between points (as opposed to travel distances)
    :param undirected:

    :return: a DGL graph
    """
    input_np = np.loadtxt(path, dtype=np.int, skiprows=7, usecols=(1,2,3))
    print("Imported file")

    # Move nodes from 1-index to 0-index. Edge weights are travel time. Unit is unknown, however the travel speed used is ~4 cm/unit.
    # Treat 1000 as the time for a travel "distance" of 1 for now
    # After transformation, weights have average 3.126, median: 2.333, std: 2.646
    row_indices, col_indices, edge_weights = input_np[:, 0]-1, input_np[:, 1]-1, input_np[:, 2]/1000

    dim = np.max(input_np[:, 0:2])

    input_mx = scipy.sparse.coo_matrix((edge_weights, (row_indices, col_indices)), shape=(dim, dim))
    g = dgl.from_scipy(input_mx)

    print("Parsed graph")

    if undirected:  # convert directed graph (as default, all the edges are directed in DGL) to undirected graph
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)

    print(g.num_nodes())
    print(g.num_edges())

    print("Converted graph")

    # Since the graph has been converted to bidirectional simple, we need to filter the original weights list, then add them
    weights_dict = {(row_indices[i], col_indices[i]): edge_weights[i] for i in range(input_np.shape[0])}
    edges1, edges2 = g.edges()
    for i in range(len(edges1)):
        edge_weights[i] = weights_dict[edges1[i].item(), edges2[i].item()]
    edge_weights = edge_weights[:len(edges1)]

    g.edata['length'] = torch.from_numpy(edge_weights)

    print("Added weights")

    if c_path != "":
        input_np_coord = np.loadtxt(c_path, dtype=np.int, skiprows=7, usecols=(2,3))/(10**6)

        g.ndata['x'] = torch.from_numpy(input_np_coord[:, 0])
        g.ndata['y'] = torch.from_numpy(input_np_coord[:, 1])

        print("Added coordinates")

    return g

def load_dgl_graph(path, undirected: bool, edge_weights=None, d_path=""):
    """
    Loads dgl graph from file in either edgelist or gr format
    """
    if ".edgelist" in path:
        return load_edgelist_file_to_dgl_graph(path, undirected, edge_weights)
    elif ".gr" in path:
        return load_gr_file_to_dgl_graph(path, undirected, d_path)
    else:
        raise ValueError('Unknown file type. Must be either .edgelist or .gr')

def download_networkx_graph(query, type, path=""):
    """
    Downloads a networkx graph from osmnx using the given query
    """
    # If not provided, assume the save path is a combination of the query and type in the data folder
    if path == "":
        path = "../data/{}-{}.pkl".format(query, type)
    if os.path.isfile(path):
        G = nx.read_gpickle(path)
    else:
        G = ox.graph_from_place(query, network_type=type)
        nx.write_gpickle(G, path)
    print("Num Nodes: {}".format(G.number_of_nodes()))
    print("Num Edges: {}".format(G.number_of_edges()))
    return G

def write_file(output_path, obj):
    ## Write to file
    if output_path is not None:
        folder_path = os.path.dirname(output_path)  # create an output folder
        if not os.path.exists(folder_path):  # mkdir the folder to store output files
            os.makedirs(folder_path)
        with open(output_path, 'wb') as f:
            dill.dump(obj, f)
    return True


def read_file(path):
    with open(path, 'rb') as f:
        generator = dill.load(f)
    return generator


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def train_valid_test_split(x, y, write_train_val_test, test_size=0.2, val_size=0.2, output_path=None, file_name=None,
                           shuffle=True,
                           random_seed=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_seed,
                                                        shuffle=shuffle, stratify=y)
    val_size_to_train_size = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size_to_train_size,
                                                      random_state=random_seed, shuffle=shuffle, stratify=y_train)

    print(
        f'shapes of train: {x_train.shape, y_train.shape}, valid: {x_val.shape, y_val.shape}, test: {x_test.shape, y_test.shape}')
    datasets = dict()
    datasets["x_train"] = x_train
    datasets["y_train"] = y_train
    datasets["x_val"] = x_val
    datasets["y_val"] = y_val
    datasets["x_test"] = x_test
    datasets["y_test"] = y_test

    if write_train_val_test:
        print(f"writing train_val_test datasets for {file_name}...")
        write_file(output_path, datasets)
        print(f"Done writing train_val_test for {file_name}")

    return datasets


def remove_data_with_a_few_observations(x, y, min_observations=6):
    original_len = len(y)
    y_keep = [k for k, v in Counter(y).items() if v >= min_observations]

    mask = np.isin(y, y_keep)
    y = y[mask]
    x = x[mask]

    print('{} rows removed from the dataset'.format(original_len - len(y)))
    return x, y


def create_dataset(distance_map: Dict, embedding, node2idx, binary_operator="average"):
    """
    create data_generator.yaml in which each data point (x,y) is (the embedding of 2 nodes, its distance)
    :param distance_map: dictionary (key, value)=(landmark_node, list_distance_to_each_node_n)
    :param embedding: embedding vectors of the nodes
    :param binary_operator: ["average", "concatenation", "subtraction", "hadamard"]
    :return: return 2 arrays:  array of data and array of labels.
    """
    if binary_operator not in ["average", "concatenation", "subtraction", "hadamard"]:
        raise ValueError(f"binary_operator is not valid!: {binary_operator}")

    data_list = []
    label_list = []
    node_pairs = set()
    for landmark in tqdm(distance_map.keys()):
        distance_list = distance_map[landmark]
        for node, distance in distance_list.items():
            pair_key = tuple(sorted([node, landmark]))
            if node == landmark or distance == np.inf or pair_key in node_pairs:
                pass
            else:
                node_pairs.add(pair_key)
                if binary_operator == "average":
                    data = (embedding[node2idx[node]] + embedding[node2idx[landmark]]) / 2.0
                else:
                    # TODO: Need to implement other binary operators
                    raise ValueError(f"binary_operator is not implemented yet!: {binary_operator}")
                label = distance
                data_list.append(np.array(data))
                label_list.append(label)

    return np.array(data_list, dtype=object), np.array(label_list, dtype=np.int16)

def get_file_name(config):
    file_name = config["graph"]["name"]
    if config["graph"]["source"] == "osmnx":
        file_name += "-" + config["graph"]["download_type"]
    # if config[]["modified"]:
    #     file_name = "modified_"+file_name
    return file_name
