import os.path

import dgl

import data_helper
import landmarks
import node2vec
from data_helper import read_file


def create_train_val_test_sets(file_name, force_override=False):
    final_output_path = f"../output/datasets/{file_name}_train_val_test.pkl"

    if not force_override and os.path.isfile(final_output_path):  ## if the file already exists
        print(f"File '{file_name}' already exists, only read it back!")
        return read_file(final_output_path)

    # TODO: should keep the config in one separate file
    ##### Step 1. Read data
    ## `file_name` is an edgelist file, no extension needed
    # "small_test" is a small dataset for testing, default data should be "socfb-American75"
    file_name = "ego-facebook-original"  # "socfb-American75"  # "socfb-American75"

    ## Load input file into a DGL graph
    input_path = f"../data/{file_name}.edgelist"
    graph = data_helper.load_edgelist_file_to_dgl_graph(path=input_path, undirected=True,
                                                        edge_weights=None)

    #####  Step 2. Run Node2Vec to get the embedding
    # Node2Vec params
    args = {
        "device": "cpu",
        "embedding_dim": 128,
        "walk_length": 50,  # 80
        "window_size": 5,  # 10
        "p": 1.0,  # 0.25,
        "q": 1.0,  # 4.0,
        "num_walks": 10,
        "epochs": 5,  # 100
        "batch_size": 128,
        "learning_rate": 0.01,
    }
    # took ~6mins/epoch to get Node2Vec for "socfb-American75", using 8GB RAM, 4CPU Mac
    embedding_output_path = f"../output/embedding/{file_name}_embed.pkl"
    embedding = node2vec.run_node2vec(graph, eval_set=None, args=args, output_path=embedding_output_path)
    print("Done embedding!")

    #####  Step 3: Create labels:
    # We convert the `dgl` graph to `networkx` graph. We will use networkx for finding the shortest path
    nx_graph = dgl.to_networkx(graph)

    ## Option 1: Get a few landmark nodes randomly from the graph:
    random_seed = 2021
    num_landmarks = 150

    ## Option 2: set `num_landmarks` to `graph.num_nodes()` to make all the nodes as landmark nodes.
    ## TODO: when all nodes are landmark nodes, might need a better way to calc the distance (symmetric matrix)
    # num_landmarks = nx_graph.number_of_nodes()

    ## Get landmark nodes:
    landmark_nodes = landmarks.get_landmark_nodes(num_landmarks, nx_graph, random_seed=random_seed)

    # Get landmarks' distance: get distance of every pair (l,n), where l is a landmark node, n is a node in the graph
    landmark_distance_output = f"../output/landmarks_distance/{file_name}_dist.pkl"  # where to store the output file
    print("Calculating landmarks distance...")
    distance_map = landmarks.calculate_landmarks_distance(landmark_nodes, nx_graph,
                                                          output_path=landmark_distance_output)
    print("Done landmarks distance!")

    ## Plot the network
    # utilities.plot_nx_graph(nx_graph, file_name=file_name)

    ##### Step 4: Create datasets to train a model
    print("creating datasets...")
    x, y = data_helper.create_dataset(distance_map, embedding)
    x, y = data_helper.remove_data_with_a_few_observations(x, y)
    test_size = 0.25
    val_size = 0.15
    train_val_test_path = "../output/datasets"
    datasets = data_helper.train_valid_test_split(x, y, test_size=test_size, val_size=val_size,
                                                  output_path=train_val_test_path,
                                                  file_name=file_name)
    print("Done writing file!")
    return datasets
