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
import networkx as nx
import osmnx as ox

import models
from datasets_generator import create_train_val_test_sets
import data_helper
from routing import GraphRouter
from Trainer import Trainer
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import pandas as pd

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

def report_percentiles(df, field):
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(df.describe())
    print("90th")
    print(df.nlargest(len(df)//10, field).iloc[-1])
    print("95th")
    print(df.nlargest(len(df)//20, field).iloc[-1])
    print("99th")
    print(df.nlargest(len(df)//100, field).iloc[-1])

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

def run_astar_csv(config, name, gr, pairs, alpha=2, report_stretch=False):
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
        csv_path = "../output/routes/{}/embedding-routes-ratio{}-dim{}{}.csv".format(file_name, config["collab_filtering"]["sample_ratio"], config["collab_filtering"]["embedding_dim"], "-"+config["collab_filtering"]["measure"] if config["collab_filtering"]["measure"] != "norm" else "")
        stretch_path = "../output/routes/{}/embedding-stretches-ratio{}-dim{}{}.csv".format(file_name, config["collab_filtering"]["sample_ratio"], config["collab_filtering"]["embedding_dim"], "-"+config["collab_filtering"]["measure"] if config["collab_filtering"]["measure"] != "norm" else "")
    else:
        csv_path = "../output/routes/{}/{}-routes.csv".format(file_name, name)
        stretch_path = "../output/routes/{}/{}-stretches.csv".format(file_name, name)

    df = pd.DataFrame(routes, columns=columns)
    df['performance'] = 1-df['pathLength']/df['numVisited']
    report_percentiles(df, 'performance')
    df.to_csv(csv_path)
    
    if report_stretch:
        columns = ["source", "target", "stretch"]
        if name == "embedding":
            stretch_path = "../output/routes/{}/embedding-stretches-ratio{}-dim{}{}.csv".format(file_name, config["collab_filtering"]["sample_ratio"], config["collab_filtering"]["embedding_dim"], "-"+config["collab_filtering"]["measure"] if config["collab_filtering"]["measure"] != "norm" else "")
        else:
            stretch_path = "../output/routes/{}/{}-stretches.csv".format(file_name, name)
        df = pd.DataFrame(stretches, columns=["source", "target", "pathDistance", "trueDistance"])
        df["stretch"] = df["pathDistance"] / df["trueDistance"]
        report_percentiles(df, 'stretch')
        df.to_csv(stretch_path)      

def plot_route(gr, f_name, u, v, alpha=2):
    G = gr.graph

    route, num_visited, visited = gr.astar(u, v, weight="length", alpha=alpha)
    print(visited[:5], visited[-5:])
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
    print(path_length/2)
    bbox = ox.utils_geo.bbox_from_point((G.nodes[mid_node]['y'], G.nodes[mid_node]['x']), dist=750)

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
        if pairs_to_csv:
            run_astar_csv(config, name, gr, pairs, alpha=alpha, report_stretch=report_stretch)
        else:
            run_astar(gr, pairs, alpha=alpha)
        print()

def generate_routing_plots(config, gr, heuristics, source=None, target=None, min_dist=5000, alpha=2):
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
        plot_route(gr, plot_path+"-A*_"+name+".png", source, target, alpha=alpha)
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
  
def run_routing_embedding(config, nx_graph, embedding, test_pairs=True, plot_route=True, run_dijkstra=True, run_dist=True, pairs_to_csv=False, alpha=2, source=None, target=None, report_stretch=False):
    """
    Run routing algorithm on given graph with given heuristic and landmark method
    :param config: provide all we need in terms of parameters
    :return: ?
    """
    gr = GraphRouter(graph=nx_graph, is_symmetric=pairs_to_csv)
    norm = config["collab_filtering"]["norm"]
    
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
    nodes = []
    def embedding_heuristic(x,y):
        x, y = gr.node_to_idx[x], gr.node_to_idx[y]
        a, b = embedding[x], embedding[y]
        D = 1000*np.linalg.norm(a-b, ord=norm)
        emb_distances.append(D)
        return D
    def hyperbolic_embedding_heuristic(x, y):
        R = 6731000
        u, v = gr.node_to_idx[x], gr.node_to_idx[y]
        a, b = embedding[u], embedding[v]
        euclid_dist = np.linalg.norm(a-b, ord=norm)
        left_norm = np.linalg.norm(a, ord=norm)
        right_norm = np.linalg.norm(b, ord=norm)
        delta = np.divide(euclid_dist**2, (1-left_norm**2)*(1-right_norm**2))
        D = R*np.arccosh(1+2*delta)
        nodes.append(x)
        emb_distances.append(D)
        return D
    def spherical_embedding_heuristic(x,y):
        R = 6731000
        x, y = gr.node_to_idx[x], gr.node_to_idx[y]
        a, b = embedding[x], embedding[y]
        dot = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        D = R*np.linalg.norm(dot, ord=norm)
        emb_distances.append(D)
        return D
    def latlong_embedding_heuristic(x,y):
        R = 6731000
        x, y = gr.node_to_idx[x], gr.node_to_idx[y]
        a, b = embedding[x], embedding[y]
        d = 0.5 - np.cos(b[0]-a[0])/2 + np.cos(a[0])*np.cos(b[0]) * (1-np.cos(b[1]-a[1]))/2
        D = 2*R*np.arcsin(np.sqrt(d))
        emb_distances.append(D)
        return D
    def invdot_embedding_heuristic(x,y):
        x, y = gr.node_to_idx[x], gr.node_to_idx[y]
        a, b = embedding[x], embedding[y]
        dot = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        D = -(dot-1)*7/2
        emb_distances.append(D)
        return D   

    heuristics = {}
    if run_dijkstra:
        heuristics["dijkstra"] = gr.heuristic
    if run_dist:
        heuristics["distance"] = dist_heuristic

    if config["collab_filtering"]["measure"] == "norm":
        heuristics["embedding"] = embedding_heuristic
    elif config["collab_filtering"]["measure"] == "hyperbolic":
        heuristics["embedding"] = hyperbolic_embedding_heuristic
    elif config["collab_filtering"]["measure"] == "spherical": # spherical
        heuristics["embedding"] = spherical_embedding_heuristic
    elif config["collab_filtering"]["measure"] == "inv-dot": # spherical
        heuristics["embedding"] = invdot_embedding_heuristic
    else:
        heuristics["embedding"] = latlong_embedding_heuristic
    if test_pairs:
        test_routing_pairs(config, gr, heuristics, pairs_to_csv, alpha, report_stretch)
    if plot_route:
        generate_routing_plots(config, gr, heuristics, source=source, target=target)

    # print(nodes[:10], emb_distances[:10])
    # print(nodes[-10:], emb_distances[-10:])
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
        generate_routing_plots(config, gr, heuristics, source=source, target=target, alpha=1000)

    if run_dist:
        print("Average Distance Heuristic:", sum(real_distances) / len(real_distances))
    print("Average True Distance:", sum(true_distances) / len(true_distances))
    print()
