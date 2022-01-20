import logging
from os import path
import os.path
from datetime import datetime

import numpy as np
from numpy.core.fromnumeric import repeat
import torch.cuda
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from torch import tensor, reshape
import dgl
import osmnx as ox

import models
from datasets_generator import create_train_val_test_sets
import data_helper
from routing import GraphRouter
from Trainer import Trainer
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

def get_test_result(config, file_name, portion, seed, model):
    """
    test model on random selected pair of nodes from the graph

    :param: file_name working data graph
    :param: portion, the portion of test data out of total data
    :seed: random seed

    :return x_test,y_test

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config["dataset"]["sample_ratio"] = portion
    config["dataset"]["sample_method"] = "random"
    config["random_seed"] = seed
    config["graph"]["name"] = file_name

    data = create_train_val_test_sets(config)
    x, y = tensor(data['x_train'].astype(np.float32)), tensor(
        data['y_train'].astype(np.float32))  # just for convenience, that's for test not for train
    pred = model(x.to(device)).reshape(-1)  # use model to predict
    y = y.detach().numpy()
    pred = pred.detach().numpy()
    return accuracy_score(np.round(pred), y), mean_absolute_error(pred, y), mean_squared_error(pred,
                                                                                               y), mean_absolute_percentage_error(
        pred, y)


def round_num(scores):
    """
    round all the scores number 
    
    return a dictionary 
    """
    for score in scores:
        scores[score] = str(round(scores[score], 4))
    return scores


#####################


def run_nn(config):
    """
    Run Neural Network Model
    :param config: provide all we need in terms of parameters
    :param force_recreate_datasets:
    :param write_train_val_test:
    :return: a neural net model
    """

    now = datetime.now()

    logging.basicConfig(filename=os.path.join(config["log_path"], "running_log.log"), level=logging.INFO)
    datasets = create_train_val_test_sets(config=config)

    file_name = config["graph"]["name"]
    portion = config["dataset"]["sample_ratio"]
    method = config["dataset"]["sample_method"]

    model = models.run_neural_net(datasets, file_name)
    logging.info("run nn on " + file_name + " at " + now.strftime("%m/%d/%Y %H:%M:%S ") + "{}%".format(
        float(portion) * 100) + " " + method)
    acc, mae, mse, mre = get_test_result(config, file_name, 0.14, 824,
                                         model)  # for small graph, we can have large portion of data to test, but for large graph
    # chose smaller portion to save time
    logging.info("ACC " + str(round(acc, 4) * 100) + "%" + " ||| " + "MAE: " + str(round(mae, 4)))
    logging.info("MSE " + str(round(mse, 4)) + " ||| " + "MRE: " + str(round(mre, 4)))
    logging.info("------------------------")
    return model


def run_some_linear_models(config, force_recreate_datasets, write_train_val_test):
    """
    Testing purpose.
    :param file_name:
    :return:
    """
    log_path = config["log_path"]
    logging.basicConfig(filename=os.path.join(log_path, "running_log.log"), level=logging.INFO)

    datasets = create_train_val_test_sets(config=config, force_recreate_datasets=force_recreate_datasets,
                                          write_train_val_test=write_train_val_test)

    ##### run some linear regression models
    scores = models.run_linear_regression(datasets, use_standard_scaler=True, merge_train_val=False)

    logging.info("run_some_linear_models_test!")
    logging.info(scores)

    return True


def run_linear_model_with_under_and_over_sampling(file_name, force_recreate_datasets, write_train_val_test,
                                                  logs_path="logs", seed_random=9999):
    """
    Try to do under and over-sampling on training set before feeding through a linear regresison model
    :param file_name:
    :param force_recreate_datasets:
    :param write_train_val_test:
    :param seed_random:
    :return:
    """
    logging.basicConfig(filename=f'../output/{logs_path}/running_log.log', level=logging.INFO)

    datasets = create_train_val_test_sets(file_name, force_recreate_datasets=force_recreate_datasets,
                                          write_train_val_test=write_train_val_test)

    x_train, y_train = datasets["x_train"], datasets["y_train"]
    values, counts = np.unique(y_train, return_counts=True)

    x = int(counts[2] * 0.7)
    y = int(0.7 * x)

    ## Undersampling
    undersample_dict = {2: y, 3: x}
    under_sampler = RandomUnderSampler(sampling_strategy=undersample_dict, random_state=seed_random)  # n_jobs=15,
    x_train, y_train = under_sampler.fit_resample(x_train, y_train.astype(np.int))
    print('Frequency of distance values after undersampling', np.unique(y_train, return_counts=True))

    ## Oversampling
    minority_samples = int(0.7 * x)
    oversample_dict = {1: minority_samples, 4: minority_samples, 5: minority_samples, 6: minority_samples,
                       7: minority_samples}
    over_sampler = RandomOverSampler(sampling_strategy=oversample_dict,
                                     random_state=seed_random)
    x_train, y_train = over_sampler.fit_resample(x_train, y_train.astype(np.int))
    print('Frequency of distance values after oversampling', np.unique(y_train, return_counts=True))

    datasets["x_train"], datasets["y_train"] = x_train, y_train
    scores = models.run_linear_regression(datasets, use_standard_scaler=True, merge_train_val=False)

    logging.info(run_linear_model_with_under_and_over_sampling)
    logging.info(scores)

    return True

def run_astar(gr, pairs, alpha=2):
    sum_visited = 0
    max_visited = 0
    max_length = 0
    max_pair = None
    for i in tqdm(range(len(pairs))):
        while True:
            u, v = pairs[i]
            try:
                path, num_visited, _ = gr.astar(u, v, alpha=alpha)
                break
            except TypeError:
                pairs[i] = (np.random.choice(gr.node_list), np.random.choice(gr.node_list))
                continue
        if num_visited / len(path) > max_visited:
            max_visited = num_visited / len(path)
            max_length = len(path)
            max_pair = pairs[i]
        sum_visited += (num_visited / len(path))
    print("Average Unnecessary Visits:", sum_visited / len(pairs))
    print("Max Unnecessary Visits:", max_visited, max_length, max_pair)
    return sum_visited / len(pairs), max_visited

def run_astar_csv(config, name, gr, pairs, alpha=2):
    if name == "embedding":
        csv_path = "../output/routes/embedding-routes-ratio{}.csv".format(config["collab_filtering"]["sample_ratio"])
    else:
        csv_path = "../output/routes/{}-routes.csv".format(name)
    with open(csv_path, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["source", "target", "numVisited", "pathLength", "uniqueVisited"])
        for i in tqdm(range(len(pairs))):
            u, v = pairs[i]
            #try:
            path, num_visited, visited = gr.astar(u, v, alpha=alpha)
            #except TypeError:
            #    continue
            csvwriter.writerow([u, v, num_visited, len(path), len(set(visited))])
      

def plot_route(gr, f_name, u, v):
    G = gr.graph

    route, num_visited, visited = gr.astar(u, v, weight="length", alpha=2)
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
    bbox = ox.utils_geo.bbox_from_point((G.nodes[mid_node]['y'], G.nodes[mid_node]['x']), dist=path_length/2)

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

    node_colors = ["white" for _ in range(G.number_of_nodes())]

    for i,n in enumerate(visited_nodes.keys()):
        node_colors[gr.node_to_idx[n]] = "blue"
    for i,n in enumerate(repeat_nodes):
        node_colors[gr.node_to_idx[n]] = "orange"

    fig, ax = ox.plot.plot_graph(G, node_color=node_colors, bbox=bbox)
    fig, ax = ox.plot.plot_graph_route(G, route, route_color='red', ax=ax)
    fig.savefig(f_name)

def test_routing_pairs(config, gr, heuristics, pairs_to_csv, alpha=2):
    print("Testing pairs")
    if pairs_to_csv:
        pairs = [(gr.node_list[i], gr.node_list[j]) for i in range(len(gr.node_list)) for j in range(i+1, len(gr.node_list))]
    else:
        pairs = np.random.choice(gr.node_list, size=(config["routing_num_samples"], 2))
    for name, heuristic in heuristics.items():
        print("A* with {} heuristic".format(name))
        gr.heuristic = heuristic
        gr.distances = {}
        if pairs_to_csv:
            run_astar_csv(config, name, gr, pairs, alpha=alpha)
        else:
            run_astar(gr, pairs, alpha=alpha)
        print()

def generate_routing_plots(config, gr, heuristics, source=None, target=None, min_dist=5000):
    print("Generating plots")
    file_name = data_helper.get_file_name(config)
    plot_path = config["graph"]["plot_path"].format(name=file_name)

    if source is None and target is None:
        length = 0
        while length < min_dist:
            u, v = np.random.choice(gr.node_list), np.random.choice(gr.node_list)
            route, _, _ = gr.astar(u, v)
            length = 0
            for i in range(1, len(route)):
                length += gr.graph.edges[route[i-1], route[i], 0]['length']
        if gr.graph.nodes[u]['y'] < gr.graph.nodes[v]['y']:
            u, v = v, u
    elif source is None:
        u = np.random.choice(gr.node_list)
    elif target is None:
        v = np.random.choice(gr.node_list)

    for name, heuristic in heuristics.items():
        print("A* with {} heuristic".format(name))
        gr.distances = {}
        gr.heuristic = heuristic
        plot_route(gr, plot_path+"-A*_"+name+".png", source, target)
        print()
        
def run_routing_model(config, nx_graph, embedding, model, test_pairs=True, plot_route=True, run_dijkstra=True, run_dist=True):
    gr = GraphRouter(graph=nx_graph)
    
    real_distances = []
    def dist_heuristic(a, b):
        R = 6731000
        p = np.pi/180
        lat_a, long_a, lat_b, long_b = gr.graph.nodes[a]['y'], gr.graph.nodes[a]['x'], gr.graph.nodes[b]['y'], gr.graph.nodes[b]['x']
        
        d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
        D = 2*R*np.arcsin(np.sqrt(d))
        real_distances.append(D)
        return D

    model_distances = []
    def model_heuristic(x, y):
        x, y = gr.node_to_idx[x], gr.node_to_idx[y]
        input = np.array((embedding[x] + embedding[y]) / 2.0).reshape(1,-1)
        # Convert prediction from km to meters
        out = Trainer.predict(model, input)[0]*1000
        model_distances.append(out)
        return out

    heuristics = {}
    if run_dijkstra:
        heuristics["dijkstra"] = gr.heuristic
    if run_dist:
        heuristics["distance"] = dist_heuristic
    heuristics["model"] = model_heuristic

    if test_pairs:
        test_routing_pairs(config, gr, heuristics)
    if plot_route:
        generate_routing_plots(config, gr, heuristics)

    if run_dist:
        print("Average Distance Heuristic:", sum(real_distances) / len(real_distances))
    print("Average Model Heuristic:", sum(model_distances) / len(model_distances))
  
def run_routing_embedding(config, nx_graph, embedding, test_pairs=True, plot_route=True, run_dijkstra=True, run_dist=True, pairs_to_csv=False, source=None, target=None):
    """
    Run routing algorithm on given graph with given heuristic and landmark method
    :param config: provide all we need in terms of parameters
    :return: ?
    """
    gr = GraphRouter(graph=nx_graph)
    
    real_distances = []
    def dist_heuristic(a, b):
        R = 6731000
        p = np.pi/180
        lat_a, long_a, lat_b, long_b = gr.graph.nodes[a]['y'], gr.graph.nodes[a]['x'], gr.graph.nodes[b]['y'], gr.graph.nodes[b]['x']
        
        d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
        D = 2*R*np.arcsin(np.sqrt(d))
        real_distances.append(D)
        return D

    emb_distances = []
    def embedding_heuristic(x,y):
        x, y = gr.node_to_idx[x], gr.node_to_idx[y]
        a, b = embedding[x], embedding[y]
        D = 1000*np.linalg.norm(a-b)
        emb_distances.append(D)
        return D

    heuristics = {}
    if run_dijkstra:
        heuristics["dijkstra"] = gr.heuristic
    if run_dist:
        heuristics["distance"] = dist_heuristic
    heuristics["embedding"] = embedding_heuristic

    if test_pairs:
        test_routing_pairs(config, gr, heuristics, pairs_to_csv)
    if plot_route:
        generate_routing_plots(config, gr, heuristics, source=source, target=target)

    if run_dist:
        print("Average Distance Heuristic:", sum(real_distances) / len(real_distances))
    print("Average Embedding Heuristic:", sum(emb_distances) / len(emb_distances))

    # if config["graph"]["source"] == "gr" or config["graph"]["source"] == "osmnx":

def run_routing_test_alphas(config, nx_graph, embedding):
    node_to_idx = {v: i for i,v in enumerate(list(nx_graph.nodes()))}
    def embedding_heuristic(x,y):
        x, y = node_to_idx[x], node_to_idx[y]
        a, b = embedding[x], embedding[y]
        D = 1000*np.linalg.norm(a-b)
        return D

    def dist_heuristic(a, b):
        R = 6731000
        p = np.pi/180
        lat_a, long_a, lat_b, long_b = nx_graph.nodes[a]['y'], nx_graph.nodes[a]['x'], nx_graph.nodes[b]['y'], nx_graph.nodes[b]['x']
        
        d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
        D = 2*R*np.arcsin(np.sqrt(d))
        return D

    gr = GraphRouter(graph=nx_graph)
    node_list = list(nx_graph.nodes())
    pairs = np.random.choice(node_list, size=(config["routing_num_samples"], 2)) # [(np.random.choice(node_list), np.random.choice(node_list)) for i in range(config["routing_num_samples"])]

    run_astar(gr, pairs)

    alphas = [1, 1.25, 1.5, 2, 5, 10]
    model_avg_visits = []
    model_max_visits = []
    dist_avg_visits = []
    dist_max_visits = []

    for alpha in alphas:
        print(alpha)
        gr.heuristic = embedding_heuristic
        gr.distances = {}
        avg_visits, max_visits = run_astar(gr, pairs, alpha)
        model_avg_visits.append(avg_visits)
        model_max_visits.append(max_visits)

        gr.heuristic = dist_heuristic
        gr.distances = {}
        avg_visits, max_visits = run_astar(gr, pairs, alpha)
        dist_avg_visits.append(avg_visits)
        dist_max_visits.append(max_visits)

    print(model_avg_visits)
    print(dist_avg_visits)
    print([dist_avg_visits[i]/model_avg_visits[i] for i in range(len(model_avg_visits))])
    print()
    print(model_max_visits)
    print(dist_max_visits)
    print([dist_max_visits[i]/model_max_visits[i] for i in range(len(model_max_visits))])

def run_routing_dist_matrix(config, nx_graph, matrix, test_pairs=True, plot_route=True, run_dijkstra=True, run_dist=True, pairs_to_csv=False, source=None, target=None):
    gr = GraphRouter(graph=nx_graph)
    
    real_distances = []
    def dist_heuristic(a, b):
        R = 6731000
        p = np.pi/180
        lat_a, long_a, lat_b, long_b = gr.graph.nodes[a]['y'], gr.graph.nodes[a]['x'], gr.graph.nodes[b]['y'], gr.graph.nodes[b]['x']
        
        d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
        D = 2*R*np.arcsin(np.sqrt(d))
        real_distances.append(D)
        return D

    true_distances = []
    def matrix_heuristic(x,y):
        x, y = gr.node_to_idx[x], gr.node_to_idx[y]
        dist = matrix[x][y]
        true_distances.append(dist)
        return dist

    heuristics = {}
    if run_dijkstra:
        heuristics["dijkstra"] = gr.heuristic
    if run_dist:
        heuristics["distance"] = dist_heuristic
    heuristics["true"] = matrix_heuristic

    if test_pairs:
        test_routing_pairs(config, gr, heuristics, pairs_to_csv, alpha=1000)
    if plot_route:
        generate_routing_plots(config, gr, heuristics, source=source, target=target)

    if run_dist:
        print("Average Distance Heuristic:", sum(real_distances) / len(real_distances))
    print("Average True Distance:", sum(true_distances) / len(true_distances))
