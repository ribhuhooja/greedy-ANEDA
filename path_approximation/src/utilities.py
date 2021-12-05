import os
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
import json


def make_log_folder(log_folder_name="logs"):
    if not os.path.exists("../output"):
        os.makedirs("../output")

    log_path = os.path.join("../output", log_folder_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return True


def plot_nx_graph(nx_g: nx.Graph, fig_size: Tuple = (15, 7), options: Dict = None, file_name=None):
    """
    Help to visualize a `networkx` graph
    :param nx_g:
    :param fig_size:
    :param options:
    :param file_name:
    :return:
    """
    if options is None:
        options = {
            'node_size': 500,
            'width': 1,
            'node_color': 'gray',
        }

    plt.figure(figsize=fig_size)
    nx.draw(nx_g, **options, with_labels=True)
    plt.savefig(f'../plots/{file_name}_pic.png')
    plt.show()
    return None


def unison_shuffle_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
