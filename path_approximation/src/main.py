from pprint import pprint

from data_helper import read_yaml
from testing_functions import run_nn
from utils import make_log_folder, generate_config_list

if __name__ == '__main__':
    ## Read the config file:
    data_generator_config = read_yaml("../configs/data_generator.yaml")

    ## Make a log folder to log the experiments
    make_log_folder(log_folder_name=data_generator_config["log_path"])

    ## Make a list of configs, we'll be running the model for each config
    config_list = generate_config_list(data_generator_config)

    for i, config in enumerate(config_list):
        print(f"...running config {i + 1}/{len(config_list)}:")
        pprint(config)
        run_nn(config=config)

    # cmd = "rm -r " + "../output/nn_return/{}".format(graph_name)
    # os.system(cmd)
