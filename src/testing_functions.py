import logging
from os import path
import os.path
import sys
from datetime import datetime
import numpy as np
import math
from numpy.core.fromnumeric import repeat
import torch.cuda
from torch import tensor, reshape
import networkx as nx
import osmnx as ox

from aneda import get_distance_numpy, get_distance
import data_helper
from routing import GraphRouter

import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import pandas as pd

def greedy_search(gr, pairs, algorithm):
    print("Running greedy search with the", algorithm, "algorithm")
    if algorithm == "normal":
        pathfn = lambda u, v: gr.greedy(u, v)
    elif algorithm == "early_abort":
        pathfn = lambda u, v: gr.greedy_early_abort(u, v)
    elif algorithm == "panic_jump":
        pathfn = lambda u, v: gr.greedy_panic_jump(u, v)

    stretches = []
    failed=0
    total=0
    num_shortest_found = 0
    true_dists = []
    for i in tqdm(range(len(pairs))):
        total+=1
        u, v, true_dist = pairs[i]
        path = pathfn(u, v)
        if path != None:
            path_dist = 0
            for i in range(1, len(path)):
                path_dist += gr.graph.edges[path[i-1], path[i], 0]['length']
            
            stretches.append(path_dist/true_dist)
            true_dists.append(true_dist)
            if path_dist == true_dist:
                num_shortest_found += 1
        else:
            failed += 1
    success_rate = 1 - (failed/total)

    
    print("Success rate is", success_rate)
    print("Average stretch is", sum(stretches)/len(stretches))
    print("Percentage of shortest paths found is", num_shortest_found/len(stretches))
    print("The average true distance is", sum(true_dists)/len(true_dists))


def test_routing_pairs_greedy(config, gr, heuristics, alpha, ratio_pairs=1, greedy_algorithm="normal"):
    file_name = data_helper.get_file_name(config)
    print("Retrieving true distances for {}".format(file_name))
    path = "../output/routes/{}/true-distances.csv".format(file_name)
    if os.path.isfile(path):
        print("Reading back distances")
        pairs = pd.read_csv(path).drop(columns=["Unnamed: 0"]).to_numpy()
    else:
        dist_map = {}
        for source in tqdm(gr.node_list):
            dist_map[source] = nx.shortest_path_length(G=gr.graph, source=source, weight="length")
        pairs = [[u, v, dist] for u in dist_map.keys() for v, dist in dist_map[u].items() if u < v]
        np.random.shuffle(pairs)
       
        
        pd.DataFrame(pairs, columns=["source", "target", "dist"]).to_csv(path)



    print("Done retrieving distances")

    print("Testing pairs")
    for name, heuristic in heuristics.items():
        print("Greedy with {} heuristic".format(name))
        gr.heuristic = heuristic
        gr.distances = {}
        print("Total number of pairs:", len(pairs))
        # if there are too many pairs only take some of them
        print("Ratio of pairs used:", ratio_pairs)
        if ratio_pairs < 1:
            pairs = pairs[:int(len(pairs)*ratio_pairs)]

        print("Number of pairs:", len(pairs))

        # sanitize pairs data
        if type(pairs[0][2]) == str:
            parse_tensors_in_pairs(pairs)

        greedy_search(gr, pairs, greedy_algorithm)

        print()


# run greedy routing
def run_greedy(config, nx_graph, embedding, alpha, ratio_pairs=1, greedy_algorithm="normal"):
    """
    Run routing algorithm on given graph with given heuristic and landmark method
    :param config: provide all we need in terms of parameters
    :return: ?
    """
    gr = GraphRouter(graph=nx_graph, is_symmetric=True)
    norm = config["aneda"]["norm"]
    
    R = 6731000 / config["graph"]["max_weight"]
    p = np.pi/180
    real_distances = []
    def dist_heuristic(a, b):
        lat_a, long_a, lat_b, long_b = gr.graph.nodes[a]['y'], gr.graph.nodes[a]['x'], gr.graph.nodes[b]['y'], gr.graph.nodes[b]['x']
        d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
        D = 2*R*np.arcsin(np.sqrt(d))
        real_distances.append(D)
        return D

    emb_distances = []
    nodes = []
    def embedding_heuristic(x,y):
        x, y = gr.node_to_idx[x], gr.node_to_idx[y]
        a, b = embedding[x], embedding[y]
        D = get_distance_numpy(a, b, config["aneda"]["measure"], config["aneda"]["norm"], config["graph"]["diameter"])
        emb_distances.append(D)
        return D

    heuristics = {}

    heuristics["embedding"] = embedding_heuristic
    print("testing pairs")
    test_routing_pairs_greedy(config, gr, heuristics, alpha, ratio_pairs, greedy_algorithm=greedy_algorithm)

# evaluate how "greedy" the embedding is
def evaluate_embedding_greediness(config, nx_graph, embedding, ratio_pairs, ratio_nodes):
    gr = GraphRouter(graph=nx_graph, is_symmetric=True)
    norm = config["aneda"]["norm"]
    
    R = 6731000 / config["graph"]["max_weight"]
    p = np.pi/180
    real_distances = []
    def dist_heuristic(a, b):
        lat_a, long_a, lat_b, long_b = gr.graph.nodes[a]['y'], gr.graph.nodes[a]['x'], gr.graph.nodes[b]['y'], gr.graph.nodes[b]['x']
        d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
        D = 2*R*np.arcsin(np.sqrt(d))
        real_distances.append(D)
        return D

    emb_distances = []
    nodes = []
    def embedding_heuristic(x,y):
        x, y = gr.node_to_idx[x], gr.node_to_idx[y]
        a, b = embedding[x], embedding[y]
        D = get_distance_numpy(a, b, config["aneda"]["measure"], config["aneda"]["norm"], config["graph"]["diameter"])
        emb_distances.append(D)
        return D

    heuristics = {}

    heuristics["embedding"] = embedding_heuristic
    test_embedding_greediness(config, gr, heuristics, ratio_pairs, ratio_nodes)

def test_embedding_greediness(config, gr, heuristics, ratio_pairs, ratio_nodes):

    print("Testing embedding greediness")
    for name, heuristic in heuristics.items():
        print("Testing embedding greediness with {} heuristic".format(name))
        gr.heuristic = heuristic
        gr.distances = {}
        print()

        # Test greediness by pairs
        num_pairs = choose_two(len(gr.node_list))
        print("Total number of pairs:", num_pairs)
        print("Ratio of pairs used:", ratio_pairs)
        num_chosen_pairs = int(num_pairs*ratio_pairs)

        all_pairs = [(u, v) for u in gr.node_list for v in gr.node_list if u<v]
        np.random.shuffle(all_pairs)
        pairs = all_pairs[:num_chosen_pairs]

        print("Number of pairs:", num_chosen_pairs)
        print()
    
        greedy_pairs = 0
        total_pairs = 0
        for i in tqdm(range(len(pairs))):
            total_pairs += 1
            first, second = pairs[i]
            if gr.pair_is_greedy(first, second):
                greedy_pairs += 1
        
        print("Percentage of greedy pairs is", 100*greedy_pairs/total_pairs, "%")
        print()


        # Test greediness on single nodes
        num_nodes = len(gr.node_list)
        print("Total number of nodes:", num_nodes)
        print("Ratio of pairs used:", ratio_nodes)
        num_chosen_nodes = int(num_nodes * ratio_nodes)
        num_chosen_pairs = int(num_pairs*ratio_pairs)
        all_nodes = [u for u in gr.node_list]
        np.random.shuffle(all_nodes)
        nodes = all_nodes[:num_chosen_nodes]
        print("number of nodes:", num_chosen_nodes)

        total_nodes = 0
        greedy_nodes = 0
        for i in tqdm(range(len(nodes))):
            total_nodes += 1
            node = nodes[i]
            is_greedy = True
            for other in gr.node_list:
                if other == node:
                    continue

                if not gr.node_is_greedy(node, other):
                    is_greedy = False
                    break
            if is_greedy:
                greedy_nodes += 1

        print("Percentage of greedy (non local-minima) nodes is", 100*greedy_nodes/total_nodes, "%")
        print()

def choose_two(n):
    return n*(n-1)//2


def parse_tensors_in_pairs(pairs):
    for i in range(len(pairs)):
        pairs[i] = (pairs[i][0], pairs[i][1], parse_float_from_tensor_string(pairs[i][2]))

def parse_float_from_tensor_string(tensor_string):
    # this is super hacky
    # the tensor string looks like 'tensor(num)'
    # we only want num
    # so string[7:-1] should get it
    return float(tensor_string[7:-1])  #eww
