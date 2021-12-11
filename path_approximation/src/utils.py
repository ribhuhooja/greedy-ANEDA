import os
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.model_selection import ParameterGrid
import copy


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


def generate_config_list(config: Dict) -> List[Dict]:
    """
    Return a list of all possible combinations of configs.
    We read the YAML file as a config, then use that config to set up the experiments.
    For some params, their values are a list (for example, when we want to tune #epochs, we can set #epochs=[10,20,30])
    We create a distinct config for each combination.

    :param config: the config read from YAML file
    :return: a list of configs
    """

    def get_tuning_params(config: Dict) -> Dict:
        parse_tuning_param_by = config["parse_tuning_param_by"]
        """
        return params we need to tune and their values
        :param config:
        :return:
        """
        tuning_params = dict()
        for config_type, config_value in config.items():
            if (isinstance(config_value, dict)):
                for k, v in config_value.items():
                    if isinstance(v, list):
                        tuning_params[f"{config_type}{parse_tuning_param_by}{k}"] = v
            else:
                if isinstance(config_value, list):
                    tuning_params[f"{config_type}"] = config_value
        return tuning_params

    def update_dict(config: Dict, params: Dict) -> Dict:
        """
        Update a `config` dict using `params`
        :param config:
        :param params:
        :return:
        """
        parse_tuning_param_by = config["parse_tuning_param_by"]

        for param_k, param_v in params.items():
            key_list = param_k.split(parse_tuning_param_by)
            if len(key_list) == 1:
                config[key_list[0]] = param_v
            else:
                config[key_list[0]][key_list[1]] = param_v
        print(id(config))
        return config

    tuning_params = get_tuning_params(config)
    tuning_params_grid = ParameterGrid(tuning_params)
    config_list = []
    for i, params in enumerate(tuning_params_grid):
        new_config = update_dict(copy.deepcopy(config), params)
        config_list.append(new_config)
    print(f"Create {len(config_list)} different configs!")
    return config_list
