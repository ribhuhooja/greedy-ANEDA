import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from metrics import mean_absolute_percentage_error


def run_linear_regression(datasets, use_standard_scaler=False, merge_train_val=True):
    x_train, y_train = datasets["x_train"], datasets["y_train"]
    x_test, y_test = datasets["x_test"], datasets["y_test"]

    if merge_train_val:
        x_val, y_val = datasets["x_val"], datasets["y_val"]
        x_train = np.vstack((x_train, x_val))
        y_train = np.hstack((y_train, y_val))

    if use_standard_scaler:
        normalize = False  # subtracting the mean and dividing by the l2-norm
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    else:
        normalize = True

    print("X.train: ", x_train.shape)
    print("X.train[0]: ", x_train[0])
    linear_regression_model = LinearRegression(fit_intercept=True, normalize=normalize, n_jobs=-1).fit(x_train, y_train)
    y_pred = linear_regression_model.predict(x_test)
    y_class = np.round(y_pred)
    linear_regression_acc = accuracy_score(y_test, y_class) * 100
    linear_regression_mse = mean_squared_error(y_test, y_pred)
    linear_regression_mae = mean_absolute_error(y_test, y_pred)
    linear_regression_mre = mean_absolute_percentage_error(y_test, y_pred)

    scores = {"acc": linear_regression_acc, "mse": linear_regression_mse, "mae": linear_regression_mae,
              "mre": linear_regression_mre}

    print("linear_regression: Accuracy={}%, MSE={}, MAE={}, MRE={}".format(round(linear_regression_acc, 3),
                                                                           round(linear_regression_mse, 3),
                                                                           round(linear_regression_mae, 3),
                                                                           round(linear_regression_mre, 3)))

    return scores


def run_neural_net():
    ## TODO:
    ## Should start off with whatever we already had
    ## Refer to https://github.com/kryptokommunist/path-length-approximation-deep-learning/blob/master/src/trainer.py
    pass

