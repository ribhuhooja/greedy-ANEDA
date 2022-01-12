import osmnx as ox
import numpy as np
from data_helper import download_networkx_graph

def real_distance(nx_graph, a, b):
    R = 6731000
    p = np.pi/180
    lat_a, long_a, lat_b, long_b = nx_graph.nodes[a]['y']*p, nx_graph.nodes[a]['x']*p, nx_graph.nodes[b]['y']*p, nx_graph.nodes[b]['x']*p
    d = 0.5 - np.cos(lat_b-lat_a)/2 + np.cos(lat_a)*np.cos(lat_b) * (1-np.cos(long_b-long_a))/2
    D = 2*R*np.arcsin(np.sqrt(d))
    return D

def approx_distance(nx_graph, a, b):
    R = 6731000
    p = np.pi/180
    lat_a, long_a, lat_b, long_b = nx_graph.nodes[a]['y']*p, nx_graph.nodes[a]['x']*p, nx_graph.nodes[b]['y']*p, nx_graph.nodes[b]['x']*p
    x = (long_b - long_a) * np.cos( 0.5*(lat_a+lat_b) )
    y = lat_b-lat_a
    d = R*np.sqrt(x*x + y*y)
    return d

def vector_distance(nx_graph, a, b):
    R = 6731000
    p = np.pi/180
    lat_a, long_a, lat_b, long_b = nx_graph.nodes[a]['y']*p, nx_graph.nodes[a]['x']*p, nx_graph.nodes[b]['y']*p, nx_graph.nodes[b]['x']*p
    vec_a = R*np.hstack((np.cos(lat_a)*np.cos(long_a), np.cos(lat_a)*np.sin(long_a), np.sin(lat_a)))
    vec_b = R*np.hstack((np.cos(lat_b)*np.cos(long_b), np.cos(lat_b)*np.sin(long_b), np.sin(lat_b)))
    return np.linalg.norm(vec_b-vec_a)

def vector_distance1(nx_graph, a, b):
    R = 6731000
    p = np.pi/180
    lat_a, long_a = (nx_graph.nodes[a]['y']+90)*p, (nx_graph.nodes[a]['x']+180)*p
    lat_b, long_b = (nx_graph.nodes[b]['y']+90)*p, (nx_graph.nodes[b]['x']+180)*p
    vec_a = R*np.hstack((np.sin(lat_a)*np.cos(long_a), np.sin(lat_a)*np.sin(long_a), np.cos(lat_a)))
    vec_b = R*np.hstack((np.sin(lat_b)*np.cos(long_b), np.sin(lat_b)*np.sin(long_b), np.cos(lat_b)))
    return np.linalg.norm(vec_b-vec_a)

def test(G):
    a, b = np.random.choice(list(G.nodes)), np.random.choice(list(G.nodes))
    real_dist = real_distance(G, a, b)
    print(real_dist)
    for dist_func in [approx_distance, vector_distance, vector_distance1]:
        dist = dist_func(G, a, b)
        print(dist, dist-real_dist)

G = download_networkx_graph("cambridge ma", "drive")
test(G)
print()
G2 = download_networkx_graph("cambridge", "drive")
test(G2)

# x = np.cos(lat)*np.cos(long + pi), 
# y = np.cos(lat)*np.sin(long + pi), 
# z = np.cos(lat + pi/2)

# theta = lat + pi/2
# azimuth = long + pi


# x = np.cos(lat)*np.cos(long)
# y = np.cos(lat)*np.sin(long)
# z = np.sin(lat)