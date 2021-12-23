import os
from datetime import datetime
from sys import path

import numpy as np
import matplotlib.pyplot as plt
import torch

import networkx as nx
import osmnx as ox
import dgl

from data_helper import read_yaml, read_file
from routing import GraphRouter
import node2vec

def get_graph(path):
    if os.path.isfile(path):
        G = nx.read_gpickle(path)
    else:
        G = ox.graph_from_place("Boston, Massachusetts, USA", network_type="walk")
        nx.write_gpickle(G, path)
    return G

def get_embeddings(path):
    dgl_graph = dgl.from_networkx(G, edge_attrs=["length"])
    print(config['node2vec']['walk_length']*dgl_graph.edata['length'].mean().item())

    if os.path.isfile(path):
        embedding = read_file(path)
        print(f"Embedding already exists! Read back from {path}")
    else:
        embedding = node2vec.run_node2vec(dgl_graph, eval_set=None, args=config['node2vec'], output_path=path)
    return embedding

def run_astar(pairs):
    sum_visited = 0
    curr_time = datetime.now()
    for i, p in enumerate(pairs):
        if i % 10 == 0:
            print(i)
        u, v = p
        _, num_visited, _ = gr.astar(u, v)
        sum_visited += num_visited
    print(datetime.now() - curr_time)
    print(sum_visited / len(pairs))

def run_routing(gr, pairs, embedding):

    print("Dijkstra's")
    run_astar(pairs)

    node_to_idx = {v: i for i,v in enumerate(list(G.nodes()))}


    def h(x,y):

        d = np.sigmoid(np.dot(embedding[node_to_idx[x]], embedding[node_to_idx[y]]))
        if d == 0:
            return 0
        D = 5000*(1-d)/d
        print(D, end=" ")
        return D

    gr.distances = {}
    gr.heuristic = h

    print("A* with DL heuristic")
    run_astar(pairs)
    

    def h1(a, b):
        R = 6731000
        p = np.pi/180
        lat_a, long_a, lat_b, long_b = gr.graph.nodes[a]['y'], gr.graph.nodes[a]['x'], gr.graph.nodes[b]['y'], gr.graph.nodes[b]['x']
        
        d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
        return 2*R*np.arcsin(np.sqrt(d))

    gr.distances = {}
    gr.heuristic = h1

    print("A* with true dist heuristic")
    run_astar(pairs)

def test_route(gr, f_name, node_to_idx, u, v):
    G = gr.graph

    route, _, visited = gr.astar(u, v, weight="length")
    path_length = 0
    for i in range(1, len(route)):
        path_length += G.edges[route[i-1], route[i], 0]['length']
    print()
    print("path length:", path_length)
    print()
    visited = set(visited)

    node_colors = [(1, 1, 1) for _ in range(G.number_of_nodes())]
    # node_colors[[node_to_idx[n] for n in visited]] = 'r'

    m = len(visited) // 6
    r, g, b = m, 0, 0
    updates = [(0,1,0),(-1,0,0),(0,0,1),(0,-1,0),(1,0,0),(0,0,-1)]
    for i,n in enumerate(visited):
        j = int((i / m) % 6)
        r, g, b = r + updates[j][0], g + updates[j][1], b + updates[j][2]
        node_colors[node_to_idx[n]] = (r/m, g/m, b/m)

    print([(r,g,b) for (r,g,b) in node_colors if r < 0 or r > 1 or g < 0 or g > 1 or b < 0 or b > 1])

    fig, ax = ox.plot.plot_graph(G, node_color=node_colors)
    fig, ax = ox.plot.plot_graph_route(G, route, route_color='black', ax=ax)
    fig.savefig(f_name)

config = read_yaml("../configs/routing.yaml")
g_path = "../data/boston-walk.pkl"
embedding_path = "../output/embedding/boston-walk2.pkl"
G = get_graph(g_path)
embedding = get_embeddings(embedding_path)

gr = GraphRouter(graph=G)
node_to_idx = {v: i for i,v in enumerate(list(G.nodes()))}
# pairs = [(np.random.choice(list(G.nodes())), np.random.choice(list(G.nodes()))) for i in range(config["routing"]["num_samples"])]
# run_routing(gr, pairs, embedding)

u = np.random.choice(list(G.nodes()))
v = np.random.choice(list(G.nodes()))

if gr.graph.nodes[u]['y'] < gr.graph.nodes[v]['y']:
    u, v = v, u

# test_route(gr, "dijkstra.png", node_to_idx, u, v)

def h(x,y):
    dot = np.dot(embedding[node_to_idx[x]], embedding[node_to_idx[y]])
    d = 1 / (1 + np.exp(-dot))
    if d == 0:
        return 0
    D = 5000*(1-d)/d
    # print(D, end=" ")
    return D

gr.distances = {}
gr.heuristic = h
test_route(gr, "A*_dl.png", node_to_idx, u, v)

def h1(a, b):
    R = 6731000
    p = np.pi/180
    lat_a, long_a, lat_b, long_b = gr.graph.nodes[a]['y'], gr.graph.nodes[a]['x'], gr.graph.nodes[b]['y'], gr.graph.nodes[b]['x']
    
    d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
    D = 2*R*np.arcsin(np.sqrt(d))
    return D

gr.distances = {}
gr.heuristic = h1
test_route(gr, "A*_dist.png", node_to_idx, u, v)

# gr = GraphRouter(graph=nx_graph)

# embedding = read_file("../output/embedding/USA-road-t.NY_embed-epochs99-lr0.01-d.pkl")
# def h(x,y):
#         # return -np.log(np.dot(model[x], model[y]))
#         d = np.dot(embedding[x], embedding[y])
#         if d == 0:
#             return 0
#         return (1-d)/d
# gr.heuristic = h
# print("A* DL:", gr.astar(u, v)[2])

# coord_table = np.loadtxt(coord_path, dtype=np.int, skiprows=7, usecols=(2,3))/(10**6)
# def h1(x, y):
#     R = 6731
#     p = np.pi/180
#     lat_x, long_x, lat_y, long_y = coord_table[x][1], coord_table[x][0], coord_table[y][1], coord_table[y][0]
    
#     a = 0.5 - np.cos((lat_y-lat_x)*p)/2 + np.cos(lat_x*p)*np.cos(lat_y*p) * (1-np.cos((long_y-long_x)*p))/2
#     return 2*R*np.arcsin(np.sqrt(a))
# gr.distances = {}
# gr.heuristic = h1
# print("A* dist:", gr.astar(u, v)[2])
