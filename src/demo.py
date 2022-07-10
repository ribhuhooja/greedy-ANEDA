import logging
from os import path
import os.path
import sys
from datetime import datetime

import numpy as np
from numpy.core.fromnumeric import repeat
import torch.cuda
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from torch import tensor, reshape

import networkx as nx
import osmnx as ox
import folium
from folium.features import ClickForMarker
import jinja2
from jinja2 import Template

from aneda import get_distance_numpy, get_distance
import data_helper
from routing import GraphRouter
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import pandas as pd

def plot_route(gr, f_name, u, v, alpha=2):
    G = gr.graph

    route, num_visited, visited = gr.astar(u, v, weight="length", alpha=alpha)
    path_length = 0
    for i in range(1, len(route)):
        path_length += G.edges[route[i-1], route[i], 0]['length']

    with open('visited.txt', 'w') as f:
        for item in visited:
            f.write("%s\n" % item)

    mid_node = None
    length_runner = 0
    for i in range(1, len(route)):
        length_runner += G.edges[route[i-1], route[i], 0]['length']
        if length_runner >= path_length // 2:
            mid_node = route[i-1]
            break
    bbox = None # ox.utils_geo.bbox_from_point((G.nodes[mid_node]['y'], G.nodes[mid_node]['x']), dist=750)

    visited_nodes = {}
    for node in visited:
        if node in visited_nodes:
            visited_nodes[node] += 1
        else:
            visited_nodes[node] = 1
    repeat_nodes = []
    for node, count in visited_nodes.items():
        if count > 1:
            repeat_nodes.append(node)

    print("Route Length:", path_length)
    print("Number of nodes in path:", len(route))
    print("Num visited:", num_visited)
    print("Unique nodes visited:", len(visited_nodes.keys()))

    node_colors = ["lightsteelblue" for _ in range(G.number_of_nodes())]

    # for i,n in enumerate(visited_nodes.keys()):
    #     node_colors[gr.node_to_idx[n]] = "darkblue"
    # for i,n in enumerate(repeat_nodes):
    #     node_colors[gr.node_to_idx[n]] = "orange"

    # map = ox.folium.plot_graph_folium(G, color='#a3dbff', weight=1.5)
    # map = ox.folium.plot_route_folium(G, route, route_map=map, color='blue')
    # folium.features.ClickForMarker().add_to(map)

    # el = folium.MacroElement().add_to(map)
    # el._template = jinja2.Template("""
    #     {% macro script(this, kwargs) %}
    #     function myFunction() {
    #     /* Get the text field */
    #     var copyText = document.getElementById("myInput");

    #     /* Select the text field */
    #     copyText.select();
    #     copyText.setSelectionRange(0, 99999); /* For mobile devices */

    #     /* Copy the text inside the text field */
    #     document.execCommand("copy");
    #     }
    #     {% endmacro %}
    # """)

    # map.save('index.html')

def generate_routing_plots(config, gr, heuristics, source=None, target=None, min_dist=1, alpha=1):
    print("Generating plots")
    file_name = data_helper.get_file_name(config)
    plot_path = config["graph"]["plot_path"].format(name=file_name)

    if source is None and target is None:
        length = 0
        while length < min_dist:
            source, target = np.random.choice(gr.node_list), np.random.choice(gr.node_list)
            res = gr.astar(source, target)
            if res is None:
                continue
            route, _, _ = res
            length = 0
            for i in range(1, len(route)):
                length += gr.graph.edges[route[i-1], route[i], 0]['length']
        if gr.graph.nodes[source]['y'] < gr.graph.nodes[target]['y']:
            source, target = target, source
    elif source is None:
        source = np.random.choice(gr.node_list)
    elif target is None:
        target = np.random.choice(gr.node_list)
    for name, heuristic in heuristics.items():
        print("A* with {} heuristic".format(name))
        gr.distances = {}
        gr.heuristic = heuristic
        plot_route(gr, plot_path+"-A*_"+name+".png", source, target, alpha=alpha)
        print()

def demo(config, nx_graph, embedding, alpha=1.5):
    """
    Run routing algorithm on given graph with given heuristic and landmark method
    :param config: provide all we need in terms of parameters
    :return: ?
    """
    gr = GraphRouter(graph=nx_graph, is_symmetric=False)
    norm = config["aneda"]["norm"]
    
    R = 6731000 / config["graph"]["max_weight"]
    p = np.pi/180
    real_distances = []
    def dist_heuristic(a, b):
        lat_a, long_a, lat_b, long_b = gr.graph.nodes[a]['y'], gr.graph.nodes[a]['x'], gr.graph.nodes[b]['y'], gr.graph.nodes[b]['x']
        d = 0.5 - np.cos((lat_b-lat_a)*p)/2 + np.cos(lat_a*p)*np.cos(lat_b*p) * (1-np.cos((long_b-long_a)*p))/2
        D = 2*R*np.arcsin(np.sqrt(d))
        real_distances.append(D)
        return D

    emb_distances = []
    def embedding_heuristic(x,y):
        x, y = gr.node_to_idx[x], gr.node_to_idx[y]
        a, b = embedding[x], embedding[y]
        D = get_distance_numpy(a, b, config["aneda"]["measure"], config["aneda"]["norm"], config["graph"]["diameter"])
        emb_distances.append(D)
        return D

    heuristics = {}

    heuristics["embedding"] = embedding_heuristic
    # generate_routing_plots(config, gr, heuristics, alpha=alpha)

    # print("Average Embedding Heuristic:", sum(emb_distances) / len(emb_distances))

    map = ox.folium.plot_graph_folium(gr.graph, color='#a3dbff', weight=1.5)
    
    ClickForMarker._customtemplate = Template(u"""
            {% macro script(this, kwargs) %}
                var markers = {};

                function newMarker(e){
                    if(Object.keys(markers).length == 2) {
                        Object.keys(markers).forEach(function(id) {
                            {{this._parent.get_name()}}.removeLayer(markers[id])
                        })
                        markers = {}
                    }
                    var new_mark = L.marker().setLatLng(e.latlng).addTo({{this._parent.get_name()}});
                    markers[new_mark._leaflet_id] = new_mark

                    new_mark.dragging.enable();
                    new_mark.on('dblclick', function(e){ 
                        delete markers[e.target._leaflet_id]
                        {{this._parent.get_name()}}.removeLayer(e.target)
                    })
                    var lat = e.latlng.lat.toFixed(4),
                       lng = e.latlng.lng.toFixed(4);
                    new_mark.bindPopup({{ this.popup }});

                    console.log(e.latlng);
                };
                {{this._parent.get_name()}}.on('click', newMarker);
            {% endmacro %}
            """)
    def __custom_init__(self, *args, **kwargs):
        self.__init_orig__(*args, **kwargs)
        self._template = self._customtemplate
    ClickForMarker.__init_orig__ = ClickForMarker.__init__
    ClickForMarker.__init__ = __custom_init__
    
    ClickForMarker().add_to(map)

    map.save('index.html')