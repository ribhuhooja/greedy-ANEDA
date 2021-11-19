import os
import dill
import numpy as np
import scipy
import dgl
from tqdm import tqdm
import networkx as nx
from typing import List, Dict
import matplotlib.pyplot as plt

def load_edgelist_file_to_dgl_graph(path: str, undirected: bool, edge_weights=None):
    """
    Reads a edgeList file in which each row contains an edge of the network, then returns a DGL graph.
    :param path: path to the edgeList file
    edgeList file  should contain 2 columns as follows:
        0 276
        0 58
        0 132

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


def calculate_landmarks_distance(landmark_nodes: List, graph: nx.Graph, output_path: Dict):
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

    ## Write to file
    if output_path is not None:
        folder_path = os.path.dirname(output_path)  # create an output folder
        if not os.path.exists(folder_path):  # mkdir the folder to store output files
            os.makedirs(folder_path)
        with open(output_path, 'wb') as f:
            dill.dump(distance_map, f)
    return distance_map


def read_pkl_file(path):
    with open(path, 'rb') as f:
        generator = dill.load(f)
    return generator

def plot_nx_graph(nx_g: nx.Graph, figsize: List = [15, 7], options: Dict = None):
    if options is None:
        options = {
            'node_color': 'black',
            'node_size': 500,
            'width': 1,
            'node_color': 'gray',
        }

    plt.figure(figsize=figsize)
    nx.draw(nx_g, **options, with_labels=True)
    plt.show()
    return None


