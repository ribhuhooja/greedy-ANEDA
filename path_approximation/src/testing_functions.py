import logging
import os.path
from datetime import datetime

import numpy as np
import torch.cuda
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from torch import tensor, reshape
from data_helper import load_edgelist_file_to_dgl_graph
import models
from datasets_generator import create_train_val_test_sets, create_node_test_pairs
import dgl

from routing import GraphRouter


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


def run_routing(config):
    """
    Run routing algorithm on given graph with given heuristic and landmark method
    :param config: provide all we need in terms of parameters
    :return: ?
    """
    now = datetime.now()

    # logging.basicConfig(filename=os.path.join(config["log_path"], f"routing_{str(now)}.log"), level=logging.INFO)
    file_name = config["data"]["file_name"]

    ##### Step 1. Read data
    ## Load input file into a DGL graph
    ## convert the `dgl` graph to `networkx` graph. We will use networkx for finding the shortest path
    input_path = config["data"]["input_path"].format(file_name=file_name)
    graph = load_edgelist_file_to_dgl_graph(path=input_path, undirected=True,
                                            edge_weights=None)

    nx_graph = dgl.to_networkx(graph)
    print(nx_graph.number_of_nodes())
    print(nx_graph.number_of_edges())

    #####  Step 2: Generate test pairs
    ## Sample landmarks, and generate a source dest pair (l, n) for every landmark l and node n
    pairs = create_node_test_pairs(nx_graph, config)

    ##### Step 3: Djikstra's
    ## Run Dijkstra's with each pair for baseline time
    gr = GraphRouter(graph=nx_graph)
    curr_time = datetime.now()
    print(len(pairs))
    indices = np.random.choice(range(len(pairs)), config["routing"]["num_samples"], replace=False)
    for idx, i in enumerate(indices):
        if idx % 50 == 0:
            print(idx)
        u, v = pairs[i]
        # print(u, v)
        _ = gr.dijkstra(u, v)
    print(datetime.now() - curr_time)

    # Generate or load embeddings and distance measure. Generate heuristic function
    # Run A* with each pair and heuristic for test time
