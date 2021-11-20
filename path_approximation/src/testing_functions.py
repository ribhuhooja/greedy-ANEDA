import models
from datasets_generator import create_train_val_test_sets


def run_some_linear_models(file_name):
    """
    Testing purpose.
    :param file_name:
    :return:
    """
    datasets = create_train_val_test_sets(file_name, force_override=False)

    ##### run some linear regression models
    models.run_linear_regression(datasets, use_standard_scaler=True, merge_train_val=False)
    # X.train:  (566,504, 128)
    # X.train[0]:  [ 2.40739295 -0.84786797 -0.60399631  1.24751753  0.09072189 -0.34991539 ...
    # linear_regression: Accuracy = 69.712 %, MSE = 0.248, MAE = 0.393, MRE = 0.16

    models.run_linear_regression(datasets, use_standard_scaler=True, merge_train_val=True)
    # X.train:  (708130, 128)
    # X.train[0]: [2.4082464 - 0.849042 - 0.60468688  1.2473787   0.08971796 - 0.34934673 ...
    # linear_regression: Accuracy=69.705%, MSE=0.248, MAE=0.393, MRE=0.16

    models.run_linear_regression(datasets, use_standard_scaler=False, merge_train_val=False)
    # X.train:  (566504, 128)
    # X.train[0]: [0.5844107866287231 - 0.15337152779102325 - 0.12185250222682953 ...
    # linear_regression: Accuracy = 69.712 %, MSE = 0.248, MAE = 0.393, MRE = 0.16

    return None
