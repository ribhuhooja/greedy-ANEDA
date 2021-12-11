import logging
import os.path
from datetime import datetime
from datasets_generator import create_train_val_test_sets, create_node_test_pairs
import data_helper
from routing import GraphRouter
import dgl
import numpy as np


def get_test_result(file_name, portion, seed, model):
    """
    test model on random selected pair of nodes from the graph

    :param: file_name working data graph
    :param: portion, the portion of test data out of total data
    :seed: random seed

    :return x_test,y_test

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = create_train_val_test_sets(file_name, True, False, portion, "random", seed)
    x, y = tensor(data['x_train'].astype(np.float32)), tensor(
        data['y_train'].astype(np.float32))  # just for convinience, that's for test not for train
    pred = [model(reshape(input_, (1, input_.size()[0])).to(device)).tolist()[0][0] for input_ in
            x]  # use model to predict
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

def run_routing(config):
    """
    Run routing algorithm on given graph with given heuristic and landmark method
    :param config: provide all we need in terms of parameters
    :return: ?
    """
    now = datetime.now()

    logging.basicConfig(filename=os.path.join(config["log_path"], "running_log.log"), level=logging.INFO)

    file_name = config["data"]["file_name"]
    portion = config["landmark"]["sample_ratio"]
    method = config["landmark"]["sample_method"]

    ##### Step 1. Read data
    ## Load input file into a DGL graph
    ## convert the `dgl` graph to `networkx` graph. We will use networkx for finding the shortest path
    input_path = config["data"]["input_path"].format(file_name=file_name)
    graph = data_helper.load_edgelist_file_to_dgl_graph(path=input_path, undirected=True,
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
    indices = np.random.choice(range(len(pairs)), 1000, replace=False)
    for idx,i in enumerate(indices):
        if idx % 50 == 0:
            print(idx)
        u, v = pairs[i]
        # print(u, v)
        _ = gr.dijkstra(u, v)
    print(datetime.now() - curr_time)

    # Generate or load embeddings and distance measure. Generate heuristic function
    # Run A* with each pair and heuristic for test time