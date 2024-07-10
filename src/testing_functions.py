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

def run_astar_csv(config, name, gr, pairs, alpha=2, report_stretch=True):
    print("The value of report stretch received by astar-csv is", report_stretch)
    file_name = data_helper.get_file_name(config)
    routes = []
    stretches = []
    for i in tqdm(range(len(pairs))):
        if report_stretch:
            u, v, true_dist = pairs[i]
        else:
            u, v = pairs[i]
        result = gr.astar(u, v, alpha=alpha)
        if result != None:
            path, num_visited, visited = result
            routes.append([u, v, num_visited, len(path), len(set(visited))])
            if report_stretch:
                path_dist = 0
                for i in range(1, len(path)):
                    path_dist += gr.graph.edges[path[i-1], path[i], 0]['length']
                stretches.append([u, v, path_dist, true_dist])

    columns = ["source", "target", "numVisited", "pathLength", "uniqueVisited"]
    if name == "embedding":
        csv_path = "../output/routes/{}/embedding-routes-ratio{}-dim{}{}.csv".format(file_name, config["aneda"]["sample_ratio"], config["aneda"]["embedding_dim"], "-"+config["aneda"]["measure"] if config["aneda"]["measure"] != "norm" else "")
        stretch_path = "../output/routes/{}/embedding-stretches-ratio{}-dim{}{}.csv".format(file_name, config["aneda"]["sample_ratio"], config["aneda"]["embedding_dim"], "-"+config["aneda"]["measure"] if config["aneda"]["measure"] != "norm" else "")
    else:
        csv_path = "../output/routes/{}/{}-routes.csv".format(file_name, name)
        stretch_path = "../output/routes/{}/{}-stretches.csv".format(file_name, name)

    df = pd.DataFrame(routes, columns=columns)
    df['performance'] = 1-df['pathLength']/df['numVisited']
    #report_percentiles(df, 'performance')
    df.to_csv(csv_path)
    
    if report_stretch:
        columns = ["source", "target", "stretch"]
        if name == "embedding":
            stretch_path = "../output/routes/{}/embedding-stretches-ratio{}-dim{}{}.csv".format(file_name, config["aneda"]["sample_ratio"], config["aneda"]["embedding_dim"], "-"+config["aneda"]["measure"] if config["aneda"]["measure"] != "norm" else "")
        else:
            stretch_path = "../output/routes/{}/{}-stretches.csv".format(file_name, name)
        df = pd.DataFrame(stretches, columns=["source", "target", "pathDistance", "trueDistance"])
        df["stretch"] = df["pathDistance"] / df["trueDistance"]
        #report_percentiles(df, 'stretch')
        df.to_csv(stretch_path)      
        print("Average stretch is", df["stretch"].mean())

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

def test_routing_pairs(config, gr, heuristics, pairs_to_csv, alpha=2, report_stretch=False):
    print("test-routing-pairs -> report stretch is", report_stretch)
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
        
# def run_routing_model(config, nx_graph, embedding, model, test_pairs=True, plot_route=True, run_dijkstra=True, run_dist=True):
#     gr = GraphRouter(graph=nx_graph)
    
#     real_distances = []
#     def dist_heuristic(a, b):
#         R = 6731000
#         p = np.pi/180
#         lat_a, long_a, lat_b, long_b = gr.graph.nodes[a]['y'], gr.graph.nodes[a]['x'], gr.graph.nodes[b]['y'], gr.graph.nodes[b]['x']
        
#         d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
#         D = 2*R*np.arcsin(np.sqrt(d))
#         real_distances.append(D)
#         return D

#     model_distances = []
#     def model_heuristic(x, y):
#         x, y = gr.node_to_idx[x], gr.node_to_idx[y]
#         input = np.array((embedding[x] + embedding[y]) / 2.0).reshape(1,-1)
#         # Convert prediction from km to meters
#         out = Trainer.predict(model, input)[0]*1000
#         model_distances.append(out)
#         return out

#     heuristics = {}
#     if run_dijkstra:
#         heuristics["dijkstra"] = gr.heuristic
#     if run_dist:
#         heuristics["distance"] = dist_heuristic
#     heuristics["model"] = model_heuristic

#     if test_pairs:
#         test_routing_pairs(config, gr, heuristics)
#     if plot_route:
#         generate_routing_plots(config, gr, heuristics)

#     if run_dist:
#         print("Average Distance Heuristic:", sum(real_distances) / len(real_distances))
#     print("Average Model Heuristic:", sum(model_distances) / len(model_distances))
  
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
        test_routing_pairs(config, gr, heuristics, pairs_to_csv, alpha, report_stretch)
    if plot_route:
        generate_routing_plots(config, gr, heuristics, source=source, target=target)

    # print(nodes[:10], emb_distances[:10])
    # print(nodes[-10:], emb_distances[-10:])
    if run_dist:
        print("Average Distance Heuristic:", sum(real_distances) / len(real_distances))
    print("Average Embedding Heuristic:", sum(emb_distances) / len(emb_distances))

    # if config["graph"]["source"] == "gr" or config["graph"]["source"] == "osmnx":


def run_routing_dist(config, nx_graph, test_pairs=True, plot_route=True, pairs_to_csv=False, alpha=1.5, source=None, target=None, report_stretch=False):
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

    heuristics = {}
    heuristics["distance"] = dist_heuristic

    if test_pairs:
        test_routing_pairs(config, gr, heuristics, pairs_to_csv, alpha, report_stretch)
    if plot_route:
        generate_routing_plots(config, gr, heuristics, source=source, target=target)

    print("Average Distance Heuristic:", sum(real_distances) / len(real_distances))

# def run_routing_test_alphas(config, nx_graph, embedding):
#     node_to_idx = {v: i for i,v in enumerate(list(nx_graph.nodes()))}
#     def embedding_heuristic(x,y):
#         x, y = node_to_idx[x], node_to_idx[y]
#         a, b = embedding[x], embedding[y]
#         D = 1000*np.linalg.norm(a-b)
#         return D

#     def dist_heuristic(a, b):
#         R = 6731000
#         p = np.pi/180
#         lat_a, long_a, lat_b, long_b = nx_graph.nodes[a]['y'], nx_graph.nodes[a]['x'], nx_graph.nodes[b]['y'], nx_graph.nodes[b]['x']
        
#         d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
#         D = 2*R*np.arcsin(np.sqrt(d))
#         return D

#     gr = GraphRouter(graph=nx_graph)
#     node_list = list(nx_graph.nodes())
#     pairs = np.random.choice(node_list, size=(config["routing_num_samples"], 2)) # [(np.random.choice(node_list), np.random.choice(node_list)) for i in range(config["routing_num_samples"])]

#     run_astar(gr, pairs)

#     alphas = [1, 1.25, 1.5, 2, 5, 10]
#     model_avg_visits = []
#     model_max_visits = []
#     dist_avg_visits = []
#     dist_max_visits = []

#     for alpha in alphas:
#         print(alpha)
#         gr.heuristic = embedding_heuristic
#         gr.distances = {}
#         avg_visits, max_visits = run_astar(gr, pairs, alpha)
#         model_avg_visits.append(avg_visits)
#         model_max_visits.append(max_visits)

#         gr.heuristic = dist_heuristic
#         gr.distances = {}
#         avg_visits, max_visits = run_astar(gr, pairs, alpha)
#         dist_avg_visits.append(avg_visits)
#         dist_max_visits.append(max_visits)

#     print(model_avg_visits)
#     print(dist_avg_visits)
#     print([dist_avg_visits[i]/model_avg_visits[i] for i in range(len(model_avg_visits))])
#     print()
#     print(model_max_visits)
#     print(dist_max_visits)
#     print([dist_max_visits[i]/model_max_visits[i] for i in range(len(model_max_visits))])

# def run_routing_dist_matrix(config, nx_graph, matrix, test_pairs=True, plot_route=True, run_dijkstra=True, run_dist=True, pairs_to_csv=False, source=None, target=None):
#     gr = GraphRouter(graph=nx_graph)
    
#     real_distances = []
#     def dist_heuristic(a, b):
#         R = 6731000
#         p = np.pi/180
#         lat_a, long_a, lat_b, long_b = gr.graph.nodes[a]['y'], gr.graph.nodes[a]['x'], gr.graph.nodes[b]['y'], gr.graph.nodes[b]['x']
        
#         d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
#         D = 2*R*np.arcsin(np.sqrt(d))
#         real_distances.append(D)
#         return D

#     true_distances = []
#     def matrix_heuristic(x,y):
#         x, y = gr.node_to_idx[x], gr.node_to_idx[y]
#         dist = matrix[x][y]
#         true_distances.append(dist)
#         return dist

#     heuristics = {}
#     if run_dijkstra:
#         heuristics["dijkstra"] = gr.heuristic
#     if run_dist:
#         heuristics["distance"] = dist_heuristic
#     heuristics["true"] = matrix_heuristic

#     if test_pairs:
#         test_routing_pairs(config, gr, heuristics, pairs_to_csv, alpha=1000)
#     if plot_route:
#         generate_routing_plots(config, gr, heuristics, source=source, target=target, alpha=1000)

#     if run_dist:
#         print("Average Distance Heuristic:", sum(real_distances) / len(real_distances))
#     print("Average True Distance:", sum(true_distances) / len(true_distances))
#     print()

def run_time_test(config, nx_graph, embedding, use_dist=True):
    pairs = np.random.choice(nx_graph.nodes(), size=(config["routing_num_samples"], 2))
    node_to_idx = {v: i for i,v in enumerate(list(nx_graph.nodes()))}

    R = 6371
    p = np.pi/180
    def dist_heuristic(a, b):
        lat_a, long_a, lat_b, long_b = nx_graph.nodes[a]['y'], nx_graph.nodes[a]['x'], nx_graph.nodes[b]['y'], nx_graph.nodes[b]['x']
        d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
        D = 2*R*np.arcsin(np.sqrt(d))
        return D

    def embedding_heuristic(x,y):
        x, y = node_to_idx[x], node_to_idx[y]
        a, b = embedding[x], embedding[y]
        D = get_distance_numpy(a, b, config["aneda"]["measure"], config["aneda"]["norm"], config["graph"]["diameter"])
        return D

    if use_dist:
        embedding_heuristic = dist_heuristic
    
    times = []
    for a,b in pairs:
        start = datetime.now()
        dist = embedding_heuristic(a, b)
        end = datetime.now()
        times.append((end-start).total_seconds())
    
    print("Average Time:", sum(times)/len(times))

    # times = []
    # dists = []
    # for i in range(1000):
    #     start = datetime.now()
    #     dist = embedding_heuristic(a, b)
    #     end = datetime.now()
    #     dists.append(dist)
    #     times.append((end-start).total_seconds())
