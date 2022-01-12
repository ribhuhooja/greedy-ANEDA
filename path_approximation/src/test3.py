import os.path
from pprint import pprint
import numpy as np
import dgl

from sklearn.decomposition import TruncatedSVD

import data_helper
from utils import make_log_folder, generate_config_list
import osmnx as ox
import matplotlib.pyplot as plt


def test_contributions(graph, embedding):
    node_list = list(graph.nodes)
    node2idx = {v:i for i,v in enumerate(node_list)}

    mean_contributions = np.zeros((1, 6))

    print("Starting test...")
    indices = np.random.choice(range(len(dataset)), size=10000)
    for i in indices:
        left, right = dataset[i, 0:2].long()
        left_emb = embedding[left.item()]
        right_emb = embedding[right.item()]
        
        contributions = (left_emb-right_emb)**2
        res = np.linalg.norm(left_emb-right_emb)**2

        if not np.isclose(res, 0):
            contributions = contributions / res
        mean_contributions += contributions/10000
    
    return mean_contributions



if __name__ == '__main__':
    ##### Here is a sample flow to run the project
    ## Firstly, change the config files in "/configs"
    ##      + data_generator.yaml: edit `file_name` to choose the input file. Pick small dataset to start first. The rest of params can be left the same
    ##      + neural_net_1.yaml: Params for neural net model
    ## Then, follow each of below steps

    ## Read the config file:
    data_generator_config = data_helper.read_yaml("../configs/collab_filtering.yaml")

    ## Make a log folder to log the experiments
    make_log_folder(log_folder_name=data_generator_config["log_path"])

    ## Make a list of configs, we'll be running the model for each config
    config_list = generate_config_list(data_generator_config)

    ## train model with a config
    for i, config in enumerate(config_list):
        pprint(config)
        file_name = data_helper.get_file_name(config)

        if config["graph"]["source"] == "osmnx":
            # OSMnx graphs have length attribute for edges, and "x" and "y" for nodes denoting longitude and latitude
            nx_graph = data_helper.download_networkx_graph(config["graph"]["name"], config["graph"]["download_type"])
            # The node attributes don't need to be kept for dgl since they won't be used in node2vec training,
            # however the edge weights are used in the modified version
            dgl_graph = dgl.from_networkx(nx_graph, edge_attrs=["length"])
        else:
            # For .gr files, distance and coordinate attributes are imported with the same names as OSMnx for consistency
            # .edgelist files are unweighted and have no spatial data, so the coordinates path with be ignored
            input_path = config["graph"]["file_path"].format(file_name=file_name)
            coord_path= config["graph"]["coord_path"].format(file_name=file_name)
            dgl_graph = data_helper.load_dgl_graph(path=input_path, undirected=True, c_path=coord_path)
            nx_graph = dgl.to_networkx(dgl_graph, edge_attrs=['length'])
        
        node_list = list(nx_graph.nodes)

        collab_filtering_args = config["collab_filtering"]

        embedding_output_path = "../output/embedding/{name}_embed-epochs{epochs}-lr{lr}-ratio{ratio}-d{dim}.pkl".format(name="collab_filtering/"+file_name,
                                                                            epochs=collab_filtering_args["epochs"],
                                                                            lr=collab_filtering_args["lr"],
                                                                            ratio=collab_filtering_args["sample_ratio"],
                                                                            dim=collab_filtering_args["embedding_dim"])
        if os.path.isfile(embedding_output_path):
            embedding = data_helper.read_file(embedding_output_path)
            print(f"Embedding already exists! Read back from {embedding_output_path}")

        dataset_output_path = "../output/datasets/collab_filtering_{}_ratio-{}".format(file_name, collab_filtering_args["sample_ratio"])

        if os.path.isfile(dataset_output_path):
            dataset = data_helper.read_file(dataset_output_path)
            print(f"Dataset already exists! Read back from {dataset_output_path}")
        np.set_printoptions(suppress=True)

        init_embedding = data_helper.get_coord_embedding(nx_graph, list(nx_graph.nodes))
        init_means = np.mean(init_embedding, axis=0)
        print(init_means)

        r = (np.linalg.norm(embedding, axis=1) / 6731).reshape(-1, 1)
        embedding = embedding / r

        print(np.mean(embedding, axis=0))
        print(np.min(embedding, axis=0))
        print(np.max(embedding, axis=0))
        print(embedding[3000]-embedding[5000], 1000*np.linalg.norm(embedding[3000]-embedding[5000]))
        print()

        embedding[:, :3] = embedding[:, :3] - init_means

        # n = collab_filtering_args["embedding_dim"]-3
        # c = np.sign(np.mean(embedding, axis=0))
        # svd = TruncatedSVD(n_components=1)
        # embedding = np.hstack((
        #     c[0]*svd.fit_transform(np.hstack((embedding[:, 0:1], embedding[:, 3:4]))),
        #     c[1]*svd.fit_transform(np.hstack((embedding[:, 1:2], embedding[:, 4:5]))),
        #     c[2]*svd.fit_transform(np.hstack((embedding[:, 2:3], embedding[:, 5:6]))),
        # ))

        def f(a, b):
            return np.sign(a+b)*np.sqrt(a*a + b*b)

        embedding = np.hstack((
            f(embedding[:, 0:1], embedding[:, 3:4]),
            f(embedding[:, 1:2], embedding[:, 4:5]),
            f(embedding[:, 2:3], embedding[:, 5:6]),
        ))
        
        embedding[:, :3] = embedding[:, :3] + init_means
        r = (np.linalg.norm(embedding, axis=1) / 6731).reshape(-1, 1)
        embedding = embedding / r
        print(np.mean(embedding, axis=0))
        print(np.min(embedding, axis=0))
        print(np.max(embedding, axis=0))
        print(embedding[3000]-embedding[5000], 1000*np.linalg.norm(embedding[3000]-embedding[5000]))
        print()

        # (x1-y1)^2 + (x2-y2)^2 = x1^2 - 2x1y1 + y1^2 + x2^2 - 
        # (f(x1, x2)-f(y1,y2))^2 = f(x1,x2)^2 - 2f(x1, x2)(y1, y2) + f(y1, y2)^2

        exit()

        fig, ax = plt.subplots(1, 2, figsize=(12, 8))

        nodes, edges = ox.graph_to_gdfs(nx_graph)
        nodes.plot(ax=ax[0], facecolor='blue', markersize=1)
        ax[0].title.set_text('Cambridge MA Original Map')

        r = np.ones(embedding.shape)
        r[:,:3] = np.linalg.norm(embedding[:3], axis=1)
        embedding = embedding / r

        p = 180/np.pi
        lat = np.arcsin(embedding[:, 2])*p
        long = np.arctan2(embedding[:, 1], embedding[:, 0])*p
        # print(np.mean(lat), np.mean(long))
        # print(np.min(lat), np.min(long))
        # print(np.max(lat), np.max(long))
        # print()

        # svd = TruncatedSVD(n_components=3)
        # embedding = svd.fit_transform(embedding)

        # r = np.linalg.norm(embedding, axis=1)
        # embedding = embedding / np.expand_dims(r, axis=1)

        # p = 180/np.pi
        # lat = np.arcsin(embedding[:, 2])*p
        # long = np.arctan2(embedding[:, 1], embedding[:, 0])*p
        # print(np.mean(lat), np.mean(long))
        # print(np.min(lat), np.min(long))
        # print(np.max(lat), np.max(long))

        for i,node in enumerate(node_list):
            nx_graph.nodes[node]['y'] = lat[i]
            nx_graph.nodes[node]['x'] = long[i]

        nodes, edges = ox.graph_to_gdfs(nx_graph)
        nodes.plot(ax=ax[1], facecolor='blue', markersize=0.3)
        ax[1].title.set_text('Cambridge MA Modified Map')

        fig.savefig("./foo.png")
        
        
# x = np.cos(lat_a)*np.cos(long_a)
# y = np.cos(lat_a)*np.sin(long_a)
# z = np.sin(lat_a)




