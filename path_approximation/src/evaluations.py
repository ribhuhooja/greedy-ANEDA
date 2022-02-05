import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from metrics import mean_absolute_percentage_error


def evaluate_metrics(y_true, y_pred, print_out, n_rounds=4, config=None):
    """
    Evaluate how well the model's doing
    :param y_true: ground truth
    :param y_pred: model's prediction
    :param print_out:
    :param n_rounds: Use round(x, `n_rounds`) to return a floating point number that is a rounded
    :return:
    """
    r = 1
    if config["graph"]["name"] == "Winterthur, Switzerland":
        r = 3.919
    elif config["graph"]["name"] == "surat":
        r = 6.99160409516
    elif config["graph"]["name"] == "ego-facebook-original":
        r = 1
        y_pred = np.round(y_pred)
    y_class = np.round(y_pred*1000)
    y_true_class = np.round(y_true*1000)
    linear_regression_acc = accuracy_score(y_true_class, y_class) * 100
    linear_regression_mse = mean_squared_error(y_true, y_pred)
    linear_regression_mae = mean_absolute_error(y_true, y_pred) / r
    linear_regression_mre = mean_absolute_percentage_error(y_true, y_pred)

    if print_out:
        print("Accuracy={}%, MSE={}, MAE={}, MRE={}".format(round(linear_regression_acc, 3),
                                                            round(linear_regression_mse, 3),
                                                            round(linear_regression_mae, 3),
                                                            round(linear_regression_mre, 3)))
    scores = {"acc": linear_regression_acc, "mse": linear_regression_mse, "mae": linear_regression_mae,
              "mre": linear_regression_mre}

    if n_rounds:
        scores = {k: round(v, n_rounds) for k, v in scores.items()}
    return scores
