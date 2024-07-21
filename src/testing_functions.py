import logging
from os import path
import os.path
import sys
from datetime import datetime
import numpy as np
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

def run_astar(gr, pairs, alpha=2):
    sum_visited = 0
    max_visited = 0
    max_length = 0
    max_pair = None
    for i in tqdm(range(len(pairs))):
        while True:
            u, v = pairs[i]
            result = gr.astar(u, v, alpha=alpha)
            if result is None:
                pairs[i] = (np.random.choice(gr.node_list), np.random.choice(gr.node_list))
                continue
            else:
                path, num_visited, _ = result
                break
        if num_visited / len(path) > max_visited:
            max_visited = num_visited / len(path)
            max_length = len(path)
            max_pair = pairs[i]
        sum_visited += (num_visited / len(path))
    print("Average Unnecessary Visits:", sum_visited / len(pairs))
    print("Max Unnecessary Visits:", max_visited, max_length, max_pair)
    return sum_visited / len(pairs), max_visited

def run_greedy_stretch_nocsv(config, name, gr, pairs, alpha=2):
    stretches = []
    failed=0
    total=0
    true_dists = []
    for i in tqdm(range(len(pairs))):
        total+=1
        u, v, true_dist = pairs[i]
        path = gr.greedy(u, v, alpha=alpha)
        if path != None:
            path_dist = 0
            for i in range(1, len(path)):
                path_dist += gr.graph.edges[path[i-1], path[i], 0]['length']
            
            stretches.append(path_dist/true_dist)
            true_dists.append(true_dist)
        else:
            failed += 1
    success_rate = 1 - (failed/total)
    
    print("Success rate is", success_rate)
    print("Average stretch is", sum(stretches)/len(stretches))
    print("The average true distance is", sum(true_dists)/len(true_dists))

def run_greedy_earlyabort_stretch_nocsv(config, name, gr, pairs, alpha=2):
    stretches = []
    failed=0
    total=0
    true_dists = []
    for i in tqdm(range(len(pairs))):
        total+=1
        u, v, true_dist = pairs[i]
        path = gr.greedy_early_abort(u, v, alpha=alpha)
        if path != None:
            path_dist = 0
            for i in range(1, len(path)):
                path_dist += gr.graph.edges[path[i-1], path[i], 0]['length']
            
            stretches.append(path_dist/true_dist)
            true_dists.append(true_dist)
        else:
            failed += 1
    success_rate = 1 - (failed/total)
    
    print("Success rate is", success_rate)
    print("Average stretch is", sum(stretches)/len(stretches))
    print("The average true distance is", sum(true_dists)/len(true_dists))


def plot_route(gr, f_name, u, v, alpha=2):
    G = gr.graph

    route, num_visited, visited = gr.astar(u, v, weight="length", alpha=alpha)
    path_length = 0
    for i in range(1, len(route)):
        path_length += G.edges[route[i-1], route[i], 0]['length']

    with open('visited.txt', 'w') as f:
        for item in visited:
            f.write("%s\n" % item)

    mid_node = None
    length_runner = 0
    for i in range(1, len(route)):
        length_runner += G.edges[route[i-1], route[i], 0]['length']
        if length_runner >= path_length // 2:
            mid_node = route[i-1]
            break
    bbox = None # ox.utils_geo.bbox_from_point((G.nodes[mid_node]['y'], G.nodes[mid_node]['x']), dist=750)

    visited_nodes = {}
    for node in visited:
        if node in visited_nodes:
            visited_nodes[node] += 1
        else:
            visited_nodes[node] = 1
    repeat_nodes = []
    for node, count in visited_nodes.items():
        if count > 1:
            repeat_nodes.append(node)

    print("Route Length:", path_length)
    print("Number of nodes in path:", len(route))
    print("Num visited:", num_visited)
    print("Unique nodes visited:", len(visited_nodes.keys()))

    node_colors = ["lightsteelblue" for _ in range(G.number_of_nodes())]

    for i,n in enumerate(visited_nodes.keys()):
        node_colors[gr.node_to_idx[n]] = "darkblue"
    # for i,n in enumerate(repeat_nodes):
    #     node_colors[gr.node_to_idx[n]] = "orange"

    fig, ax = ox.plot.plot_graph(G, node_color=node_colors, bbox=bbox, bgcolor="white", edge_color="dodgerblue")
    fig, ax = ox.plot.plot_graph_route(G, route, route_color='black', ax=ax)
    fig.savefig(f_name)

def test_routing_pairs_greedy(config, gr, heuristics, alpha, ratio_pairs=1):
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
        start = datetime.now()
        print("Running greedy_csv")
        print("Total number of pairs:", len(pairs))
        # if there are too many pairs only take some of them
        print("Ratio of pairs used:", ratio_pairs)
        if ratio_pairs < 1:
            pairs = pairs[:int(len(pairs)*ratio_pairs)]

        print("Number of pairs:", len(pairs))

        # sanitize pairs data
        if type(pairs[0][2]) == str:
            print("Sanitizing pair data")
            parse_tensors_in_pairs(pairs)


        run_greedy_earlyabort_stretch_nocsv(config, name, gr, pairs, alpha=alpha)
        end = datetime.now()
        original_stdout = sys.stdout
        with open('routes.txt', 'w') as f:
            sys.stdout = f
            print("greedy with {} heuristic".format(name), (end-start).total_seconds())
        sys.stdout = original_stdout
        print()


def test_routing_pairs(config, gr, heuristics, pairs_to_csv, alpha=2, report_stretch=False):
    if pairs_to_csv:
        if report_stretch:
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
        else:
            pairs = [(gr.node_list[i], gr.node_list[j]) for i in range(len(gr.node_list)) for j in range(i+1, len(gr.node_list))]
            np.random.shuffle(pairs)
    else:
        pairs = np.random.choice(gr.node_list, size=(config["routing_num_samples"], 2))

    print("Testing pairs")
    for name, heuristic in heuristics.items():
        print("A* with {} heuristic".format(name))
        gr.heuristic = heuristic
        gr.distances = {}
        start = datetime.now()
        if pairs_to_csv:
            print("Running astar_csv")
            run_astar_csv(config, name, gr, pairs, alpha=alpha, report_stretch=report_stretch)
        else:
            run_astar(gr, pairs, alpha=alpha)
        end = datetime.now()
        original_stdout = sys.stdout
        with open('routes.txt', 'w') as f:
            sys.stdout = f
            print("A* with {} heuristic".format(name), (end-start).total_seconds())
        sys.stdout = original_stdout
        print()

def generate_routing_plots(config, gr, heuristics, source=None, target=None, min_dist=1, alpha=1):
    print("Generating plots")
    file_name = data_helper.get_file_name(config)
    plot_path = config["graph"]["plot_path"].format(name=file_name)

    if source is None and target is None:
        length = 0
        while length < min_dist:
            source, target = np.random.choice(gr.node_list), np.random.choice(gr.node_list)
            res = gr.astar(source, target)
            if res is None:
                continue
            route, _, _ = res
            length = 0
            for i in range(1, len(route)):
                length += gr.graph.edges[route[i-1], route[i], 0]['length']
        if gr.graph.nodes[source]['y'] < gr.graph.nodes[target]['y']:
            source, target = target, source
    elif source is None:
        source = np.random.choice(gr.node_list)
    elif target is None:
        target = np.random.choice(gr.node_list)
    for name, heuristic in heuristics.items():
        print("A* with {} heuristic".format(name))
        gr.distances = {}
        gr.heuristic = heuristic
        plot_route(gr, plot_path+"-A*_"+name+".png", source, target, alpha=alpha)
        print()
        
def run_routing_embedding(config, nx_graph, embedding, test_pairs=True, plot_route=True, run_dijkstra=True, run_dist=True, pairs_to_csv=False, alpha=1.5, source=None, target=None, report_stretch=False):
    """
    Run routing algorithm on given graph with given heuristic and landmark method
    :param config: provide all we need in terms of parameters
    :return: ?
    """
    gr = GraphRouter(graph=nx_graph, is_symmetric=pairs_to_csv)
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
    if run_dijkstra:
        heuristics["dijkstra"] = gr.heuristic
    if run_dist:
        heuristics["distance"] = dist_heuristic

    heuristics["embedding"] = embedding_heuristic
    if test_pairs:
        print("testing pairs")
        test_routing_pairs(config, gr, heuristics, pairs_to_csv, alpha, report_stretch) #this is where astar is run
    if plot_route:
        generate_routing_plots(config, gr, heuristics, source=source, target=target)

    if run_dist:
        print("Average Distance Heuristic:", sum(real_distances) / len(real_distances))
    print("Average Embedding Heuristic:", sum(emb_distances) / len(emb_distances))

# run greedy routing
def run_greedy(config, nx_graph, embedding, alpha, ratio_pairs=1):
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
    test_routing_pairs_greedy(config, gr, heuristics, alpha, ratio_pairs) #this is where astar is run

# evaluate how "greedy" the embedding is
def evaluate_embedding_greediness(config, nx_graph, embedding, ratio_pairs):
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
    test_embedding_greediness(config, gr, heuristics, ratio_pairs)

def test_embedding_greediness(config, gr, heuristic, ratio_pairs):

    print("Testing embedding greediness")
    for name, heuristic in heuristics.items():
        print("Testing embedding greediness with {} heuristic".format(name))
        gr.heuristic = heuristic
        gr.distances = {}
        print()
        print("Total number of pairs:", len(pairs))
        # if there are too many pairs only take some of them
        print("Ratio of pairs used:", ratio_pairs)
        if ratio_pairs < 1:
            pairs = pairs[:int(len(pairs)*ratio_pairs)]
        print("Number of pairs:", len(pairs))
        print()
        # sanitize pairs data
        if type(pairs[0][2]) == str:
            parse_tensors_in_pairs(pairs)
    
        greedy_pairs = 0
        total_pairs = 0
        for i in tqdm(range(len(pairs))):
            total_pairs += 1
            u, v, true_dist = pairs[i]
            path = gr.greedy_early_abort(u, v, alpha=alpha)
            if path != None:
                path_dist = 0
                for i in range(1, len(path)):
                    path_dist += gr.graph.edges[path[i-1], path[i], 0]['length']
                
                stretches.append(path_dist/true_dist)
                true_dists.append(true_dist)
            else:
                failed += 1
        success_rate = 1 - (failed/total)
        
        print("Success rate is", success_rate)
        print("Average stretch is", sum(stretches)/len(stretches))
        print("The average true distance is", sum(true_dists)/len(true_dists))

        print()



def parse_tensors_in_pairs(pairs):
    for i in range(len(pairs)):
        pairs[i] = (pairs[i][0], pairs[i][1], parse_float_from_tensor_string(pairs[i][2]))

def parse_float_from_tensor_string(tensor_string):
    # this is super hacky
    # the tensor string looks like 'tensor(num)'
    # we only want num
    # so string[7:-1] should get it
    return float(tensor_string[7:-1])  #eww
