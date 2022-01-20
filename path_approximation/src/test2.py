import osmnx as ox
import numpy as np
from data_helper import download_networkx_graph

def real_distance(lat_a, long_a, lat_b, long_b):
    R = 6731000
    p = np.pi/180
    lat_a, long_a, lat_b, long_b = lat_a*p, long_a*p, lat_b*p, long_b*p
    d = 0.5 - np.cos(lat_b-lat_a)/2 + np.cos(lat_a)*np.cos(lat_b) * (1-np.cos(long_b-long_a))/2
    D = 2*R*np.arcsin(np.sqrt(d))
    return D

def approx_distance(lat_a, long_a, lat_b, long_b):
    R = 6731000
    p = np.pi/180
    lat_a, long_a, lat_b, long_b = lat_a*p, long_a*p, lat_b*p, long_b*p
    x = (long_b - long_a) * np.cos( 0.5*(lat_a+lat_b) )
    y = lat_b-lat_a
    d = R*np.sqrt(x*x + y*y)
    return d

def vector_distance(lat_a, long_a, lat_b, long_b):
    R = 6731000
    p = np.pi/180
    lat_a, long_a, lat_b, long_b = lat_a*p, long_a*p, lat_b*p, long_b*p
    vec_a = R*np.hstack((np.cos(lat_a)*np.cos(long_a), np.cos(lat_a)*np.sin(long_a), np.sin(lat_a)))
    vec_b = R*np.hstack((np.cos(lat_b)*np.cos(long_b), np.cos(lat_b)*np.sin(long_b), np.sin(lat_b)))
    return np.linalg.norm(vec_b-vec_a)

def vector_distance1(lat_a, long_a, lat_b, long_b):
    R = 6731000
    p = np.pi/180
    lat_a, long_a = (lat_a+90)*p, (long_a+180)*p
    lat_b, long_b = (lat_b+90)*p, (long_b+180)*p
    vec_a = R*np.hstack((np.sin(lat_a)*np.cos(long_a), np.sin(lat_a)*np.sin(long_a), np.cos(lat_a)))
    vec_b = R*np.hstack((np.sin(lat_b)*np.cos(long_b), np.sin(lat_b)*np.sin(long_b), np.cos(lat_b)))
    return np.linalg.norm(vec_b-vec_a)

def hyperbolic_distance(lat_a, long_a, lat_b, long_b):
    R = 6731000
    r=np.sqrt(2)-1# 1-1/np.sqrt(R)
    p = np.pi/180
    lat_a, long_a, lat_b, long_b = lat_a*p, long_a*p, lat_b*p, long_b*p
    vec_a = r*np.hstack((np.cos(lat_a)*np.cos(long_a), np.cos(lat_a)*np.sin(long_a), np.sin(lat_a)))
    vec_b = r*np.hstack((np.cos(lat_b)*np.cos(long_b), np.cos(lat_b)*np.sin(long_b), np.sin(lat_b)))
    delta = np.dot(vec_b-vec_a, vec_b-vec_a)/((1-np.dot(vec_a, vec_a))*(1-np.dot(vec_b, vec_b)))
    return R*np.arccosh(1+2*delta)

def test(G, G2):
    a, b = np.random.choice(list(G.nodes)), np.random.choice(list(G2.nodes))
    lat_a, long_a, lat_b, long_b = G.nodes[a]['y'], G.nodes[a]['x'], G2.nodes[b]['y'], G2.nodes[b]['x']
    real_dist = real_distance(lat_a, long_a, lat_b, long_b)
    print(real_dist)
    distance_functions = [("approx", approx_distance), ("vector", vector_distance), ("vector2", vector_distance1), ("hyperbolic", hyperbolic_distance)]
    for name, dist_func in distance_functions:
        dist = dist_func(lat_a, long_a, lat_b, long_b)
        print(name, dist, dist-real_dist)

G = download_networkx_graph("cambridge ma", "drive")
G2 = download_networkx_graph("waltham ma", "drive")
# G2 = download_networkx_graph("cambridge", "drive")
# G3 = download_networkx_graph("Zhouzhuang", "drive")
test(G, G)
print()
test(G, G2)
# print()
# test(G, G3)
# print()
# test(G2, G3)


# x = np.cos(lat)*np.cos(long + pi), 
# y = np.cos(lat)*np.sin(long + pi), 
# z = np.cos(lat + pi/2)

# theta = lat + pi/2
# azimuth = long + pi


# x = np.cos(lat)*np.cos(long)
# y = np.cos(lat)*np.sin(long)
# z = np.sin(lat)