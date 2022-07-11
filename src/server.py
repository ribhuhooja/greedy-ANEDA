from flask import Flask, request
from flask_cors import CORS, cross_origin

import osmnx as ox
import networkx as nx
import folium
import json
import numpy as np

from main import main
from aneda import get_distance_numpy
from routing import GraphRouter

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

class Router():
    def __init__(self, config, nx_graph, embedding):
        def embedding_heuristic(x,y):
            x, y = self.gr.node_to_idx[x], self.gr.node_to_idx[y]
            a, b = embedding[x], embedding[y]
            D = get_distance_numpy(a, b, config["aneda"]["measure"], config["aneda"]["norm"], config["graph"]["diameter"])
            return D
        self.gr = GraphRouter(graph=nx_graph, is_symmetric=False)
        self.gr.heuristic = embedding_heuristic
        self.config = config
        self.graph = nx_graph
        self.coords = [(self.graph.nodes[node]['y'], self.graph.nodes[node]['x']) for node in self.gr.node_list]

    def get_closest_node(self, lat_lng):
        lat, lng = tuple(lat_lng.split('|'))
        lat, lng = float(lat), float(lng)

        p = np.pi/180
        min_dist = 1000
        min_dist_idx = -1
        for i, (lat2,lng2) in enumerate(self.coords):
            dist = 0.5 - np.cos((lat2-lat)*p)/2 + np.cos(lat*p)*np.cos(lat2*p) * (1-np.cos((lng2-lng)*p))/2
            if dist < min_dist:
                min_dist = dist
                min_dist_idx = i
        if min_dist_idx == -1:
            raise ValueError("min_dist initialization is too small for graph")
        return self.gr.node_list[min_dist_idx]

    def get_route(self, u, v):
        route, _, _ = self.gr.astar(u, v, weight="length", alpha=1.5)

        lines = []
        node_pairs = zip(route[:-1], route[1:])
        uvk = ((u, v, min(self.graph[u][v], key=lambda k: self.graph[u][v][k]["length"])) for u, v in node_pairs)
        gdf = ox.utils_graph.graph_to_gdfs(self.graph.subgraph(route), nodes=False).loc[uvk]
        for vals in gdf[['geometry']].values:
            params = dict(zip(["geom", "popup_val"], vals))
            line = [(lat, lng) for lng, lat in params['geom'].coords]
            lines.append(line)
        return lines
            

config, nx_graph, embedding = main("../configs/routing.yaml")
print("loaded")
router = Router(config, nx_graph, embedding)
print("routed")

@app.route("/", methods=['GET'])
def route():
    src_lat_lng = request.args.get('src')
    dest_lat_lng = request.args.get('dest')

    src_node = router.get_closest_node(src_lat_lng)
    dest_node = router.get_closest_node(dest_lat_lng)
    route = router.get_route(src_node, dest_node)

    return {'route': route}

app.run()