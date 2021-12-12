import os.path

import dgl
from sklearn.model_selection import train_test_split

import data_helper
import landmarks
import node2vec
from data_helper import read_file
from utils import plot_nx_graph
import networkx as nx


def create_train_val_test_sets(config, mode):
    file_name = config["data"]["file_name"]
    random_seed = config["random_seed"]

    if mode == "train":
        final_output_path = config["data"]["final_train_val_sets_path"].format(file_name=file_name,
                                                                               sample_ratio_for_training=
                                                                               config["landmark"][
                                                                                   "sample_ratio_for_training"],
                                                                               sample_method=config["landmark"][
                                                                                   "sample_method"],
                                                                               val_size=config["val_size"])
        force_recreate = config["force_recreate_train_and_val_sets"]

    elif mode == "test":
        final_output_path = config["data"]["final_test_set_path"].format(file_name=file_name,
                                                                         sample_ratio_for_testing=config["landmark"][
                                                                             "sample_ratio_for_testing"],
                                                                         sample_method=config["landmark"][
                                                                             "sample_method"],
                                                                         )
        force_recreate = config["force_recreate_test_set"]
    else:
        raise ValueError()

    if (not force_recreate) and os.path.isfile(final_output_path):  ## if the file already exists
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

    sample_method = None

    if mode == "test":
        sample_method = 'random'
        sample_ratio = config["landmark"]["sample_ratio_for_testing"]
        random_seed = random_seed + 1  # change to a different random seed to make test set different from training sets
    elif mode == "train":
        sample_method = config["landmark"]["sample_method"]
        sample_ratio = config["landmark"]["sample_ratio_for_training"]
    else:
        raise ValueError("mode should be 'train' or 'test'!")

    if sample_method == 'random':
        num_landmarks = int(len(nx_graph) * sample_ratio)
        landmark_nodes = landmarks.get_landmark_nodes(num_landmarks, nx_graph, random_seed=random_seed)
    elif sample_method == 'high_degree':
        landmark_nodes = landmarks.get_landmark_custom(nx_graph, portion=sample_ratio)
    elif sample_method == "high_and_low_degree":
        landmark_nodes = landmarks.get_landmark_custom2(nx_graph, portion=sample_ratio)
    elif 'centrality' in sample_method:
        landmark_nodes = landmarks.get_landmark_custom3(nx_graph, portion=sample_ratio,
                                                        centrality_type=sample_method)
    elif sample_method == 'medium_degree':
        landmark_nodes = landmarks.get_landmark_custom4(nx_graph, portion=sample_ratio)

    else:
        raise ValueError(
            f"landmark sampling method should be in ['betweenness_centrality', 'closeness_centrality','random','high_degree','medium_degree'], instead of {sample_method}!")

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
    x, y = data_helper.create_dataset(distance_map, embedding)
    x, y = data_helper.remove_data_with_a_few_observations(x, y)

    if mode == "train":
        print("creating train and val sets for training...")

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=config["val_size"], random_state=random_seed,
                                                          shuffle=True, stratify=y)

        datasets = dict()
        datasets["x_train"] = x_train
        datasets["y_train"] = y_train
        datasets["x_val"] = x_val
        datasets["y_val"] = y_val

        if config["write_train_val_sets"]:
            print(f'writing train_val_sets datasets for {config["data"]["file_name"]}...')
            data_helper.write_file(final_output_path, datasets)
            print(f'Done writing train and val sets for {config["data"]["file_name"]}')

    elif mode == "test":
        print("creating a test set for testing...")
        datasets = dict()
        datasets["x_test"] = x
        datasets["y_test"] = y

        if config["write_test_set"]:
            print(f'writing test set datasets for {config["data"]["file_name"]}...')
            data_helper.write_file(final_output_path, datasets)
            print(f'Done writing test set for {config["data"]["file_name"]}')
    else:
        raise ValueError("mode should be 'train' or 'test'!")

    return datasets


def create_node_test_pairs(graph, config):
    # Sample source nodes
    random_seed = config["random_seed"]
    sample_method = config["routing"]["sample_method"]
    sample_ratio = config["routing"]["sample_ratio"]
    if sample_method == 'random':
        num_landmarks = int(len(graph) * sample_ratio)
        landmark_nodes = landmarks.get_landmark_nodes(num_landmarks, graph, random_seed=random_seed)
    elif sample_method == 'high_degree':
        landmark_nodes = landmarks.get_landmark_custom(graph, portion=sample_ratio)
    elif sample_method == "high_and_low_degree":
        landmark_nodes = landmarks.get_landmark_custom2(graph, portion=sample_ratio)
    elif 'centrality' in sample_method:
        landmark_nodes = landmarks.get_landmark_custom3(graph, portion=sample_ratio, centrality_type=sample_method)
    elif sample_method == 'medium_degree':
        landmark_nodes = landmarks.get_landmark_custom4(graph, portion=sample_ratio)
    else:
        raise ValueError(
            f"landmark sampling method should be in ['betweenness_centrality', 'closeness_centrality','random','high_degree','medium_degree'], instead of {sample_method}!")
    pairs = []
    # Generate source, dest pairs
    for landmark in landmark_nodes:
        node_dists = nx.shortest_path_length(G=graph, source=landmark)
        for node, _ in node_dists.items():
            pairs.append((landmark, node))
    return pairs
