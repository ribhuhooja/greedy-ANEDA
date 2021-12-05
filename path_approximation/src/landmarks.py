from typing import List

import networkx as nx
import numpy as np
from tqdm import tqdm

from data_helper import write_file


def get_landmark_custom(g: nx.Graph,portion):
    """
    This function pickes those nodes with hightest edges 
    """
    ## sort node accoding to degrees
    s = sorted(g.degree, key=lambda x: x[1], reverse=True)
    landmark_nodes = [node[0] for node in s[:int(len(s) * portion)]]
    return landmark_nodes

def get_landmark_custom2(g: nx.Graph,portion):
    """
    This function pickes those nodes with hightest edges(half) and lowest edges(half)
    """
    s = sorted(g.degree, key=lambda x: x[1], reverse=True)
    landmark_nodes_high = [node[0] for node in s[:int(len(s) * (portion/2))]]
    landmark_nodes_low = [node[0] for node in s[-int(len(s) * (portion/2)):]]
    return landmark_nodes_high + landmark_nodes_low
    
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
