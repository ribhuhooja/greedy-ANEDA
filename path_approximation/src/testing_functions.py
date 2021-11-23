import logging

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import models
from datasets_generator import create_train_val_test_sets

def run_nn(file_name,force_recreate_datasets,write_train_val_test,logs_path = 'logs'):
    logging.basicConfig(filename=f'../output/{logs_path}/running_log.log', level=logging.INFO)
    datasets = create_train_val_test_sets(file_name, force_recreate_datasets=force_recreate_datasets,
                                          write_train_val_test=write_train_val_test)
    scores = models.run_neural_net(datasets)
    
    logging.info("run_nn")
    logging.info(scores)

def run_some_linear_models(file_name, force_recreate_datasets, write_train_val_test, logs_path="logs"):
    """
    Testing purpose.
    :param file_name:
    :return:
    """
    logging.basicConfig(filename=f'../output/{logs_path}/running_log.log', level=logging.INFO)

    datasets = create_train_val_test_sets(file_name, force_recreate_datasets=force_recreate_datasets,
                                          write_train_val_test=write_train_val_test)

    ##### run some linear regression models
    scores = models.run_linear_regression(datasets, use_standard_scaler=True, merge_train_val=False)
    # X.train:  (566,504, 128)
    # X.train[0]:  [ 2.40739295 -0.84786797 -0.60399631  1.24751753  0.09072189 -0.34991539 ...
    # linear_regression: Accuracy = 69.712 %, MSE = 0.248, MAE = 0.393, MRE = 0.16

    # models.run_linear_regression(datasets, use_standard_scaler=True, merge_train_val=True)
    ## X.train:  (708130, 128)
    ## X.train[0]: [2.4082464 - 0.849042 - 0.60468688  1.2473787   0.08971796 - 0.34934673 ...
    ## linear_regression: Accuracy=69.705%, MSE=0.248, MAE=0.393, MRE=0.16
    #
    # models.run_linear_regression(datasets, use_standard_scaler=False, merge_train_val=False)
    # # X.train:  (566504, 128)
    # # X.train[0]: [0.5844107866287231 - 0.15337152779102325 - 0.12185250222682953 ...
    # # linear_regression: Accuracy = 69.712 %, MSE = 0.248, MAE = 0.393, MRE = 0.16
    logging.info("run_some_linear_models")
    logging.info(scores)

    return True


def run_linear_model_with_under_and_over_sampling(file_name, force_recreate_datasets, write_train_val_test,
                                                  logs_path="logs", seed_random=9999):
    """
    Try to do under and over-sampling on training set before feeding through a linear regresison model
    :param file_name:
    :param force_recreate_datasets:
    :param write_train_val_test:
    :param seed_random:
    :return:
    """
    logging.basicConfig(filename=f'../output/{logs_path}/running_log.log', level=logging.INFO)

    datasets = create_train_val_test_sets(file_name, force_recreate_datasets=force_recreate_datasets,
                                          write_train_val_test=write_train_val_test)

    x_train, y_train = datasets["x_train"], datasets["y_train"]
    values, counts = np.unique(y_train, return_counts=True)

    x = int(counts[2] * 0.7)
    y = int(0.7 * x)

    ## Undersampling
    undersample_dict = {2: y, 3: x}
    under_sampler = RandomUnderSampler(sampling_strategy=undersample_dict, random_state=seed_random)  # n_jobs=15,
    x_train, y_train = under_sampler.fit_resample(x_train, y_train.astype(np.int))
    print('Frequency of distance values after undersampling', np.unique(y_train, return_counts=True))

    ## Oversampling
    minority_samples = int(0.7 * x)
    oversample_dict = {1: minority_samples, 4: minority_samples, 5: minority_samples, 6: minority_samples,
                       7: minority_samples}
    over_sampler = RandomOverSampler(sampling_strategy=oversample_dict,
                                     random_state=seed_random)
    x_train, y_train = over_sampler.fit_resample(x_train, y_train.astype(np.int))
    print('Frequency of distance values after oversampling', np.unique(y_train, return_counts=True))

    datasets["x_train"], datasets["y_train"] = x_train, y_train
    scores = models.run_linear_regression(datasets, use_standard_scaler=True, merge_train_val=False)

    logging.info(run_linear_model_with_under_and_over_sampling)
    logging.info(scores)

    return True
