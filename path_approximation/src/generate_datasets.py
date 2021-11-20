import dgl
import node2vec
import utilities

def main():
    # TODO: should keep the config in one separate file

    ##### Step 1. Read data
    ## Input file: should be an edgelist file, no extension needed
    # "small_test" is a small dataset for testing, default data should be "socfb-American75"
    file_name = "small_test"  # "socfb-American75"

    ## Load input file into a DGL graph
    input_path = f"../data/{file_name}.edgelist"
    graph = utilities.load_edgelist_file_to_dgl_graph(path=input_path, undirected=True,
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
    num_landmarks = 20  # 150

    ## Option 2: set `num_landmarks` to `graph.num_nodes()` to make all the nodes as landmark nodes.
    ## TODO: when all nodes are landmark nodes, need a better way to calc the distance (symmetric matrix)
    # num_landmarks = nx_graph.number_of_nodes()

    ## Get landmark nodes:
    landmark_nodes = utilities.get_landmark_nodes(num_landmarks, nx_graph, random_seed=random_seed)

    # Get landmarks' distance: get distance of every pair (l,n), where l is a landmark node, n is a node in the graph
    landmark_distance_output = f"../output/landmarks_distance/{file_name}_dist.pkl"  # where to store the output file
    print("Calculating landmarks distance...")
    distance_map = utilities.calculate_landmarks_distance(landmark_nodes, nx_graph,
                                                          output_path=landmark_distance_output)
    print("Done landmarks distance!")

    ## Plot the network
    utilities.plot_nx_graph(nx_graph, file_name=file_name)

    ##### Step 4: Create datasets to train a model
    print("creating datasets...")
    x, y = utilities.create_dataset(distance_map, embedding)
    output_path = "../output/datasets"
    utilities.get_train_valid_test_split(x, y, output_path=output_path, file_name=file_name)
    print("Done writing file!")

    ##### Step 5: Create model
    # Refer to https://github.com/kryptokommunist/path-length-approximation-deep-learning/blob/master/src/trainer.py

    ##### Step 6: Evaluate the results

if __name__ == '__main__':
    main()

