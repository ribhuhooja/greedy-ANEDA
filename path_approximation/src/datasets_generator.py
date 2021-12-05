import os.path

import dgl

import data_helper
import landmarks
import node2vec
from data_helper import read_file
from utilities import plot_nx_graph


def create_train_val_test_sets(config, force_recreate_datasets, write_train_val_test):
    file_name = config["data"]["file_name"]
    random_seed = config["random_seed"]

    final_output_path = config["data"]["final_train_val_test_path"].format(file_name=file_name)
    if (not force_recreate_datasets) and os.path.isfile(final_output_path):  ## if the file already exists
        print(f"datasets for '{file_name}' already exists, only read them back!")
        return read_file(final_output_path)

    ##### Step 1. Read data
    ## Load input file into a DGL graph
    input_path = config["data"]["input_path"].format(file_name=file_name)
    graph = data_helper.load_edgelist_file_to_dgl_graph(path=input_path, undirected=True,
                                                        edge_weights=None)

    ##### Step 2. Run Node2Vec to get the embedding
    node2vec_args = config["node2vec"]
    embedding_output_path = config["data"]["embedding_output_path"].format(file_name=file_name,
                                                                           epochs=node2vec_args["epochs"],
                                                                           lr=node2vec_args["lr"])
    if os.path.isfile(embedding_output_path):
        embedding = read_file(embedding_output_path)
        print(f"Embedding already exists! Read back from {embedding_output_path}")
    else:
        embedding = node2vec.run_node2vec(graph, eval_set=None, args=node2vec_args, output_path=embedding_output_path)
    print(f"Done embedding {file_name}!")

    #####  Step 3: Create labels
    # convert the `dgl` graph to `networkx` graph. We will use networkx for finding the shortest path
    nx_graph = dgl.to_networkx(graph)
    sample_method = config["landmark"]["sample_method"]
    sample_ratio = config["landmark"]["sample_ratio"]
    if sample_method == 'random':
        num_landmarks = int(len(nx_graph) * sample_ratio)
        landmark_nodes = landmarks.get_landmark_nodes(num_landmarks, nx_graph, random_seed=random_seed)
    elif sample_method == 'high_degree':
        landmark_nodes = landmarks.get_landmark_custom(nx_graph, portion=sample_ratio)
    else:
        raise ValueError(f"landmark sampling method should be in [random, high_degree], instead of {sample_method}!")

    # TODO: when all nodes are landmark nodes, might need a better way to calc the distance (symmetric matrix)

    # Get landmarks' distance: get distance of every pair (l,n), where l is a landmark node, n is a node in the graph
    landmark_distance_output_path = config["data"]["landmark_distance_output_path"].format(file_name=file_name)
    print("Calculating landmarks distance...")
    distance_map = landmarks.calculate_landmarks_distance(landmark_nodes, nx_graph,
                                                          output_path=landmark_distance_output_path)
    print("Done landmarks distance!")

    ## Plot the network
    if config["plot"]["plot_nx_graph"]:
        plot_nx_graph(nx_graph, file_name=file_name)

    ##### Step 4: Create datasets to train a model
    print("creating datasets...")
    x, y = data_helper.create_dataset(distance_map, embedding)
    x, y = data_helper.remove_data_with_a_few_observations(x, y)
    test_size = config["train_val_test"]["test_size"]
    val_size = config["train_val_test"]["val_size"]
    datasets = data_helper.train_valid_test_split(x, y, test_size=test_size, val_size=val_size,
                                                  output_path=final_output_path,
                                                  file_name=file_name, write_train_val_test=write_train_val_test,
                                                  random_seed=random_seed)
    return datasets
