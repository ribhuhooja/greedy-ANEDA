import logging
import os.path
from datetime import datetime

import numpy as np
import torch.cuda
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from torch import tensor, reshape
from data_helper import load_file_to_dgl_graph, read_file, write_file
import models
from datasets_generator import create_train_val_test_sets, create_node_test_pairs
import dgl

from routing import GraphRouter
from Trainer import Trainer


def get_test_result(config, file_name, portion, seed, model):
    """
    test model on random selected pair of nodes from the graph

    :param: file_name working data graph
    :param: portion, the portion of test data out of total data
    :seed: random seed

    :return x_test,y_test

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config["landmark"]["sample_ratio"] = portion
    config["landmark"]["sample_method"] = "random"
    config["random_seed"] = seed
    config["data"]["file_name"] = file_name

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

    file_name = config["data"]["file_name"]
    portion = config["landmark"]["sample_ratio"]
    method = config["landmark"]["sample_method"]

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


def run_routing(config, graph, model):
    """
    Run routing algorithm on given graph with given heuristic and landmark method
    :param config: provide all we need in terms of parameters
    :return: ?
    """
    now = datetime.now()

    # logging.basicConfig(filename=os.path.join(config["log_path"], f"routing_{str(now)}.log"), level=logging.INFO)
    file_name = config["data"]["file_name"]

    # ##### Step 1. Read data
    # ## Load input file into a DGL graph
    # ## convert the `dgl` graph to `networkx` graph. We will use networkx for finding the shortest path
    # print("Started loading")
    # input_path = config["data"]["input_path"].format(file_name=file_name)
    # graph = load_file_to_dgl_graph(path=input_path, undirected=True, d_path=config["routing"]["dist_path"])
    # print("Finished loading")

    if ".edgelist" in config["data"]["input_path"]:
        nx_graph = dgl.to_networkx(graph, edge_attrs=['weight'])
    elif ".gr" in config["data"]["input_path"]:
        nx_graph = dgl.to_networkx(graph, edge_attrs=['weight']) #, 'dist'])

    # print(nx_graph.number_of_nodes())
    # print(nx_graph.number_of_edges())

    #####  Step 2: Generate test pairs
    ## Sample landmarks, and generate a source dest pair (l, n) for every landmark l and node n
    # pairs = create_node_test_pairs(nx_graph, config)

    ##### Step 3: Djikstra's
    ## Run Dijkstra's with each pair for baseline time
    gr = GraphRouter(graph=nx_graph)
    # print(len(pairs))
    # indices = np.random.choice(range(nx_graph.number_of_edges()), config["routing"]["num_samples"], replace=False)
    pairs = []

    print("Dijkstra's")
    sum_visited = 0
    curr_time = datetime.now()
    for i in range(config["routing"]["num_samples"]):
    # for idx, i in enumerate(indices):
        if i % 5 == 0:
            print(i)
        # u, v = pairs[i]
        u = np.random.choice(range(nx_graph.number_of_nodes()))
        v = np.random.choice(range(nx_graph.number_of_nodes()))
        pairs.append((u,v))
        # print(u, v)
        # _ = gr.dijkstra(u, v)
        # _, num_visited = gr.astar(u, v)
        # sum_visited += num_visited
    print(datetime.now() - curr_time)
    print(sum_visited / config["routing"]["num_samples"])


    # Load embeddings and distance measure. Generate heuristic function
    file_name = config["data"]["file_name"]
    node2vec_args = config["node2vec"]
    embedding_output_path = config["data"]["embedding_output_path"].format(file_name=file_name,
                                                                           epochs=node2vec_args["epochs"],
                                                                           lr=node2vec_args["lr"])
    # embedding = read_file(embedding_output_path)

    def h(x,y):
        # return -np.log(np.dot(model[x], model[y]))
        d = np.dot(model[x], model[y])
        if d == 0:
            return 0
        return (1-d)/d

    gr.distances = {}
    if ".edgelist" in config["data"]["input_path"]:
        # gr.heuristic = lambda x, y: Trainer.predict(model, np.array((embedding[x] + embedding[y]) / 2.0).reshape(1,-1))[0]
        gr.heuristic = h #-np.log(np.dot(model[x], model[y]))
    elif ".gr" in config["data"]["input_path"]:
        gr.heuristic = h # lambda x, y: gr.graph.get_edge_data(x, y)['dist']

    # predict(model: nn.Module, x: Union[np.array, torch.Tensor])

    # Run A* with each pair and heuristic for test time
    # print(len(gr.distances.keys()))
    print("A* with DL heuristic")
    curr_time = datetime.now()
    sum_visited = 0
    for idx, p in enumerate(pairs):
        if idx % 5 == 0:
            print(idx)
        u, v = p # pairs[i]
        # print(u, v)
        _, num_visited = gr.astar(u, v)
        sum_visited += num_visited
    print(datetime.now() - curr_time)
    print(sum_visited / len(pairs))
    # write_file("./distances.json", gr.distances)

    coord_table = np.loadtxt(config["routing"]["coord_path"], dtype=np.int, skiprows=7, usecols=(2,3))/(10**6)

    def h1(x, y):
        R = 6731000
        p = np.pi/180
        lat_x, long_x, lat_y, long_y = coord_table[x][1], coord_table[x][0], coord_table[y][1], coord_table[y][0]
        
        a = 0.5 - np.cos((lat_y-lat_x)*p)/2 + np.cos(lat_x*p)*np.cos(lat_y*p) * (1-np.cos((long_y-long_x)*p))/2
        return 2*R*np.arcsin(np.sqrt(a))

    gr.distances = {}
    if ".gr" in config["data"]["input_path"]:
        gr.heuristic = h1

    # predict(model: nn.Module, x: Union[np.array, torch.Tensor])

    # Run A* with each pair and heuristic for test time
    # print(len(gr.distances.keys()))
    print("A* with true dist heuristic")
    curr_time = datetime.now()
    sum_visited = 0
    for idx, p in enumerate(pairs):
        if idx % 5 == 0:
            print(idx)
        u, v = p # pairs[i]
        # print(u, v)
        # _, num_visited = gr.astar(u, v)
        # sum_visited += num_visited
    print(datetime.now() - curr_time)
    print(sum_visited / len(pairs))
    # write_file("./distances.json", gr.distances)
