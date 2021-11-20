import os
from typing import List, Dict, Tuple

import dgl
import dill
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def write_file(output_path, obj):
    ## Write to file
    if output_path is not None:
        folder_path = os.path.dirname(output_path)  # create an output folder
        if not os.path.exists(folder_path):  # mkdir the folder to store output files
            os.makedirs(folder_path)
        with open(output_path, 'wb') as f:
            dill.dump(obj, f)
    return True


def load_edgelist_file_to_dgl_graph(path: str, undirected: bool, edge_weights=None):
    """
    Reads a edgeList file in which each row contains an edge of the network, then returns a DGL graph.
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
    return g


def get_landmark_nodes(num_landmarks: int, graph: nx.Graph, random_seed: int = None) -> List:
    """
    Given a graph, return `num_landmarks` random nodes in the graph.
    If  `num_landmarks` >= num of nodes, return all the nodes in the graph as landmark nodes
    :param num_landmarks:
    :param graph: a networkx graph as we use networkx  for finding the shortest path
    :param random_seed:
    :return: a list of random nodes in the graph
    """

    if num_landmarks >= graph.number_of_nodes():
        return list(graph.nodes)  ## get all nodes as landmark nodes

    if random_seed is not None:
        ## Set random seed
        np.random.seed(random_seed)

    ## Pick random nodes from the graph to make them as landmark nodes:
    landmark_nodes = np.random.choice(range(graph.number_of_nodes()), num_landmarks, replace=False)
    return landmark_nodes


def calculate_landmarks_distance(landmark_nodes: List, graph: nx.Graph, output_path: str):
    """
    Calculate the distance between each landmark node `l` to a node `n` in the graph
    :param landmark_nodes:
    :param graph:
    :param output_path:
    :return: a dict containing distance from each landmark node `l` to every node in the graph
    """

    nodes = list(graph.nodes)

    distance_map = {}
    distances = np.zeros((len(nodes),))

    for landmark in tqdm(landmark_nodes):
        distances[:] = np.inf
        node_dists = nx.shortest_path_length(G=graph, source=landmark)
        for node_n, dist_to_n in node_dists.items():
            distances[node_n] = dist_to_n

        distance_map[landmark] = distances.copy()

    write_file(output_path, distance_map)
    return distance_map


def read_pkl_file(path):
    with open(path, 'rb') as f:
        generator = dill.load(f)
    return generator


def plot_nx_graph(nx_g: nx.Graph, fig_size: Tuple = (15, 7), options: Dict = None, file_name=None):
    if options is None:
        options = {
            'node_size': 500,
            'width': 1,
            'node_color': 'gray',
        }

    plt.figure(figsize=fig_size)
    nx.draw(nx_g, **options, with_labels=True)
    plt.savefig(f'../plots/{file_name}_pic.png')
    plt.show()
    return None


def create_dataset(distance_map: Dict, embedding, binary_operator="average"):
    """
    create dataset in which each data point (x,y) is (the embedding of 2 nodes, its distance)
    :param distance_map: dictionary (key, value)=(landmark_node, list_distance_to_each_node_n)
    :param embedding: embedding vectors of the nodes
    :param binary_operator: ["average", "concatenation", "subtraction", "hadamard"]
    :return: return 2 arrays:  array of data and  array of labels.
    """
    if binary_operator not in ["average", "concatenation", "subtraction", "hadamard"]:
        raise ValueError(f"binary_operator is not valid!: {binary_operator}")

    data_list = []
    label_list = []
    node_pairs = set()
    for landmark in distance_map.keys():
        distance_list = distance_map[landmark]
        for node, distance in enumerate(tqdm(distance_list)):
            pair_key = tuple(sorted([node, landmark]))
            if node == landmark or distance == np.inf or pair_key in node_pairs:
                pass
            else:
                node_pairs.add(pair_key)
                if binary_operator == "average":
                    data = (embedding[node] + embedding[landmark]) / 2.0
                else:
                    # TODO: Need to implement other binary operators
                    raise ValueError(f"binary_operator is not implemented yet!: {binary_operator}")
                label = distance
                data_list.append(np.array(data))
                label_list.append(label)

    return np.array(data_list, dtype=object), np.array(label_list, dtype=np.int16)


def get_train_valid_test_split(x, y, test_size=0.2, val_size=0.2, output_path=None, file_name=None, shuffle=True, random_seed=None):
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
    write_file(os.path.join(output_path, f"{file_name}_train_val_test.pkl"), datasets)

    return datasets
