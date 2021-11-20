from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from data_helper import write_file


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
    :return: return 2 arrays:  array of data and array of labels.
    """
    if binary_operator not in ["average", "concatenation", "subtraction", "hadamard"]:
        raise ValueError(f"binary_operator is not valid!: {binary_operator}")

    data_list = []
    label_list = []
    node_pairs = set()
    for landmark in tqdm(distance_map.keys()):
        distance_list = distance_map[landmark]
        for node, distance in enumerate(distance_list):
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
