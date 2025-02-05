import copy
import math
import time
from evaluations import evaluate_metrics

import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils import data as torch_data
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from Trainer import Trainer
from data_helper import read_yaml
from metrics import mean_absolute_percentage_error

from utils import *


def run_linear_regression(datasets, use_standard_scaler=False, merge_train_val=True):
    print("training linear regression model...")
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

    # print("X.train: ", x_train.shape)
    # print("X.train[0]: ", x_train[0])
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


def run_neural_net(datasets, file_name):
    ## Refer to https://github.com/kryptokommunist/path-length-approximation-deep-learning/blob/master/src/trainer.py

    params = read_yaml("../configs/neural_net_1.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scores = {}
    # return datasets
    x_train, y_train = datasets["x_train"].astype(np.float32), datasets["y_train"].astype(np.float32)
    x_valid, y_valid = datasets["x_val"].astype(np.float32), datasets['y_val'].astype(np.float32)
    x_test, y_test = datasets["x_test"].astype(np.float32), datasets["y_test"].astype(np.float32)
    params['input_size'] = x_train.shape[1]

    # x_train,y_train = unison_shuffle_copies(x_train, y_train)
    trainset = torch_data.TensorDataset(torch.as_tensor(x_train, dtype=torch.float, device=device),
                                        torch.as_tensor(y_train, dtype=torch.float, device=device))
    train_dl = torch_data.DataLoader(trainset, batch_size=params['batch_size'], drop_last=True)
    val_dl = torch_data.DataLoader(torch_data.TensorDataset(torch.as_tensor(x_valid, dtype=torch.float, device=device),
                                                            torch.as_tensor(y_valid, dtype=torch.float, device=device)),
                                   batch_size=params['batch_size'], drop_last=True)
    test_dl = torch_data.DataLoader(torch_data.TensorDataset(torch.as_tensor(x_test, dtype=torch.float, device=device),
                                                             torch.as_tensor(y_test, dtype=torch.float, device=device)),
                                    batch_size=params['batch_size'], drop_last=True)

    values, counts = np.unique(y_train, return_counts=True)
    max_dist = math.ceil(max(values))  # have questions: why it is a float

    def ensure_folders_exist(path):
        """
        Creates folders that do not exist yet on the given path
        """
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:  # Guard against race condition
                raise

    path = f"../output/nn_return/{file_name}"
    ensure_folders_exist(path)

    def get_test_scores(model, best_model_path, test_dl, loss_fn, writer, lrs):
        print("Saving test scores")

        def test(model, dl):
            model.eval()
            final_loss = 0.0
            count = 0
            y_hat = []
            with torch.no_grad():
                for data_cv in dl:
                    inputs, dist_true = data_cv[0], data_cv[1]
                    count += len(inputs)
                    outputs = model(inputs)
                    y_hat.extend(outputs.tolist())
                    loss = loss_fn(outputs, dist_true)
                    final_loss += loss.item()
            return final_loss / len(dl), y_hat

        if best_model_path:
            model.load_state_dict(torch.load(best_model_path))
        test_loss, y_hat = test(model, test_dl)
        print(test_loss)
        writer.add_text('test-loss', str(test_loss))

        acc_score = accuracy_score(y_test[:len(y_hat)], np.round(y_hat))

        writer.add_text('Accuracy=', str(acc_score))
        print(str(accuracy_score(y_test[:len(y_hat)], np.round(y_hat))))

        y_hat_ = np.array(y_hat).squeeze()
        y_test_ = y_test[:len(y_hat)]
        print(len(y_test), len(y_hat))
        dist_accuracies = []
        dist_mae = []
        dist_mre = []
        dist_mse = []
        dist_counts = []

        for i in range(max_dist + 1):
            mask = y_test_ == i
            dist_values = y_test_[mask]
            dist_preds = np.round(y_hat_[mask])
            if len(dist_values) != len(dist_preds):
                print(
                    "ERROR: len(dist_values) != len(dist_preds) => {} != {}".format(len(dist_values), len(dist_preds)))
            elif len(dist_values) < 1 and i != 0:
                dist_accuracies.append(np.nan)
                dist_mae.append(np.nan)
                dist_mse.append(np.nan)
                dist_mre.append(np.nan)
                dist_counts.append(len(dist_values))
                continue
            elif len(dist_values) < 1 and i == 0:
                continue
            dist_accuracies.append(np.sum(dist_values == dist_preds) * 100 / len(dist_values))
            dist_mae.append(mean_absolute_error(dist_values, dist_preds))
            dist_mse.append(mean_squared_error(dist_values, dist_preds))
            dist_mre.append(mean_absolute_percentage_error(dist_values, dist_preds))
            dist_counts.append(len(dist_values))
        mse = mean_squared_error(np.array(y_hat).squeeze(), y_test[:len(y_hat)])
        mae = mean_absolute_error(np.array(y_hat).squeeze(), y_test[:len(y_hat)])
        mre = mean_absolute_percentage_error(np.array(y_hat).squeeze(), y_test[:len(y_hat)])

        result = {"acc": acc_score, "mse": mse, "mae": mae, "mre": mre, "lr_arr": lr_arr[:len(lrs)],
                  "dist_accuracies": dist_accuracies, "dist_mae": dist_mae, "dist_mre": dist_mre,
                  "dist_mse": dist_mse, "dist_counts": dist_counts}
        return result

    def poisson_loss(y_pred, y_true):
        """
        Custom loss function for Poisson model.
        Equivalent Keras implementation for reference:
        K.mean(y_pred - y_true * math_ops.log(y_pred + K.epsilon()), axis=-1)
        For output of shape (2,3) it return (2,) vector. Need to calculate
        mean of that too.
        """
        y_pred = torch.squeeze(y_pred)
        loss = torch.mean(y_pred - y_true * torch.log(y_pred + 1e-7))
        return loss

    def get_model():
        """
        creates a PyTorch model. Change the 'params' dict above to
        modify the neural net configuration.
         -> CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC

        https://github.com/keras-team/keras/issues/1802
        http://torch.ch/blog/2016/02/04/resnets.html
        https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        """

        model = torch.nn.Sequential(
            torch.nn.Linear(params['input_size'], params['hidden_units_1']),
            torch.nn.BatchNorm1d(params['hidden_units_1']),
            # torch.nn.Dropout(p=params['do_1']),
            # torch.nn.ReLU(),
            torch.nn.Softplus(),
            torch.nn.Linear(params['hidden_units_1'], params['hidden_units_2']),
            torch.nn.BatchNorm1d(params['hidden_units_2']),
            # torch.nn.Dropout(p=params['do_2']),
            # torch.nn.ReLU(),
            torch.nn.Softplus(),
            torch.nn.Linear(params['hidden_units_2'], params['hidden_units_3']),
            torch.nn.BatchNorm1d(params['hidden_units_3']),
            # torch.nn.Dropout(p=params['do_3']),
            # torch.nn.ReLU(),
            torch.nn.Softplus(),
            torch.nn.Linear(params['hidden_units_3'], params['output_size']),
            # torch.nn.ReLU(),
            torch.nn.Softplus(),
        )
        model.to(device)
        return model

    model = get_model()
    print('model loaded into device=', next(model.parameters()).device)
    summary(model, input_size=(params['input_size'],), device=device)

    lr_reduce_patience = 20
    lr_reduce_factor = 0.1

    loss_fn = poisson_loss
    # optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=params['lr'], alpha=0.99, eps=1e-08, weight_decay=0,
                                    momentum=0, centered=False)

    if params['lr_sched'] == 'reduce_lr_plateau':
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduce_factor,
                                                              patience=lr_reduce_patience, verbose=True,
                                                              threshold=0.00001, threshold_mode='rel', cooldown=0,
                                                              min_lr=1e-9, eps=1e-08)
    elif params['lr_sched'] == 'clr':
        lr_sched = torch.optim.lr_scheduler.CyclicLR(optimizer, params['min_lr'], params['max_lr'],
                                                     step_size_up=8 * len(train_dl), step_size_down=None,
                                                     mode=params['lr_sched_mode'], last_epoch=-1, gamma=params['gamma'])

    print('lr scheduler type:', lr_sched)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    global lr_arr
    lr_arr = np.zeros((len(train_dl),))

    ################################################################

    def find_lr(init_value=1e-8, final_value=10., beta=0.98):
        global lr_arr
        num = len(train_dl) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        lrs = []
        for data in train_dl:
            batch_num += 1
            # As before, get the loss for this mini-batch of inputs/outputs
            inputs, labels = data
            # inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            lrs.append(lr)
            lr_arr[batch_num - 1] = lr
            # Do the SGD step
            loss.backward()
            optimizer.step()
            # Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr
        return log_lrs, losses

    lrs, losses = find_lr()
    print('returned', len(losses))
    plt.figure()
    plt.plot(lr_arr[:len(lrs)], losses)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title('LR range plot')
    plt.xlabel('Learning rates')
    plt.ylabel('Losses')
    im_path = os.path.join(path, "lr range plot.png")
    ensure_folders_exist(im_path)
    fig, ax = plt.subplots()
    fig.savefig(im_path)

    print(" --- optimizer.param_groups[0]['lr'] : ", optimizer.param_groups[0]['lr'])

    def evaluate(model, dl):
        """
        This function is used to evaluate the model with validation.
        args: model and data loader
        returns: loss
        """
        model.eval()
        final_loss = 0.0
        count = 0
        with torch.no_grad():
            for data_cv in dl:
                inputs, dist_true = data_cv[0], data_cv[1]
                count += len(inputs)
                outputs = model(inputs)
                loss = loss_fn(outputs, dist_true)
                final_loss += loss.item()
        return final_loss / len(dl)

    def save_checkpoint(state, state_save_path):
        if not os.path.exists("/".join(state_save_path.split('/')[:-1])):
            os.makedirs("/".join(state_save_path.split('/')[:-1]))
        torch.save(state, state_save_path)

    last_loss = 0.0
    min_val_loss = np.inf
    patience_counter = 0
    early_stop_patience = 50
    best_model = None
    train_losses = []
    val_losses = []

    tb_path = os.path.join(path, 'logs/runs')
    checkpoint_path = os.path.join(tb_path, 'checkpoints')
    resume_training = False
    start_epoch = 0
    iter_count = 0
    #
    # if os.path.exists(checkpoint_path):
    #     # raise Exception("this experiment already exists!")
    #     print("Already ran training on {}".format(checkpoint_path))
    #     return

    ensure_folders_exist(checkpoint_path)

    writer = SummaryWriter(log_dir=tb_path, comment='', purge_step=None, max_queue=1, flush_secs=30, filename_suffix='')
    writer.add_graph(model, input_to_model=torch.zeros(params['input_size'], device=device).view(1, -1),
                     verbose=False)  # not useful

    # resume training on a saved model
    if resume_training:
        prev_checkpoint_path = '../outputs/logs/runs/run42_clr_g0.95/checkpoints'  # change this
        suffix = '1592579305.7273214'  # change this
        model.load_state_dict(torch.load(prev_checkpoint_path + '/model_' + suffix + '.pt'))
        optimizer.load_state_dict(torch.load(prev_checkpoint_path + '/optim_' + suffix + '.pt'))
        lr_sched.load_state_dict(torch.load(prev_checkpoint_path + '/sched_' + suffix + '.pt'))
        state = torch.load(prev_checkpoint_path + '/state_' + suffix + '.pt')
        start_epoch = state['epoch']
        writer.add_text('loaded saved model:', str(params))
        print('loaded saved model', params)

    writer.add_text('run_change', 'Smaller 3 hidden layer NN, no DO' + str(params))

    torch.backends.cudnn.benchmark = True
    print('total epochs=', len(range(start_epoch, start_epoch + params['epochs'])))

    # with torch.autograd.detect_anomaly():  # use this to detect bugs while training
    for param_group in optimizer.param_groups:
        print('lr-check', param_group['lr'])

    epoch_wise_scores = []
    for epoch in range(start_epoch, start_epoch + params['epochs']):  # loop over the dataset multiple times
        running_loss = 0.0
        stime = time.time()

        try:
            print("optimizer.param_groups[0]['lr']: ", optimizer.param_groups[0]['lr'])
            print("lr_sched.get_last_lr(): ", lr_sched.get_last_lr())
        except:
            print("can not print out learning rate")

        for i, data in enumerate(train_dl, 0):
            iter_count += 1
            # get the inputs; data is a list of [inputs, dist_true]
            model.train()
            inputs, dist_true = data[0], data[1]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, dist_true)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            last_loss = loss.item()

            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
            writer.add_scalar('monitor/lr-iter', curr_lr, iter_count - 1)

            if not isinstance(lr_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # print("if not isinstance(lr_sched, torch.optim.lr_scheduler.ReduceLROnPlateau): ",
                #       lr_sched.get_last_lr())

                lr_sched.step()

        val_loss = evaluate(model, val_dl)
        print("val_loss: ", val_loss)

        validation_loss, metric_scores = Trainer().evaluate_model(model, loss_fn, val_dl, device,
                                                                  evaluate_function=evaluate_metrics)
        print("validation_loss: ", validation_loss)
        print("metric_scores: ", metric_scores)

        print("=======")

        if isinstance(lr_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_sched.step(val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
            best_model = copy.deepcopy(model)
            print(epoch, "> Best val_loss model saved:", round(val_loss, 4))
        else:
            patience_counter += 1
        train_loss = running_loss / len(train_dl)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)

        epoch_wise_scores.append(get_test_scores(model, False, test_dl, loss_fn, writer, lrs))

        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
        writer.add_scalar('monitor/lr-epoch', curr_lr, epoch)
        if patience_counter > early_stop_patience:
            print("Early stopping at epoch {}. current val_loss {}".format(epoch, val_loss))
            break

        """
        if epoch % 10 == 0:
            torch.save(best_model.state_dict(), os.path.join(checkpoint_path, 'model_cp.pt'))
            torch.save(optimizer.state_dict(), checkpoint_path+'/optim_cp.pt')
            torch.save(lr_sched.state_dict(), checkpoint_path+'/sched_cp.pt')
            writer.add_text('checkpoint saved', 'at epoch='+str(epoch))
        """

        print("epoch:{} -> train_loss={},val_loss={} - {}".format(epoch, round(train_loss, 5), round(val_loss, 5),
                                                                  (time.time() - stime)))

    print('Finished Training')
    ts = str(time.time())
    best_model_path = os.path.join(checkpoint_path, 'model_' + ts + '.pt')
    opt_save_path = os.path.join(checkpoint_path, 'optim_' + ts + '.pt')
    sched_save_path = os.path.join(checkpoint_path, 'sched_' + ts + '.pt')
    state_save_path = os.path.join(checkpoint_path, 'state_' + ts + '.pt')
    state = {'epoch': epoch + 1,
             'model_state': model.state_dict(),
             'optim_state': optimizer.state_dict(),
             'last_train_loss': train_losses[-1],
             'last_val_loss': val_losses[-1],
             'total_iters': iter_count
             }

    save_checkpoint(state, state_save_path)
    # sometimes loading from state dict is not wokring, so...
    torch.save(best_model.state_dict(), best_model_path)
    torch.save(optimizer.state_dict(), opt_save_path)
    torch.save(lr_sched.state_dict(), sched_save_path)

    scores["nn"] = get_test_scores(model, best_model_path, test_dl, loss_fn, writer, lrs)
    scores["nn"]["lr_losses"] = losses
    scores["nn"]["train_losses"] = train_losses
    scores["nn"]["val_losses"] = val_losses
    scores["nn"]["epoch_wise_scores"] = epoch_wise_scores

    writer.add_text('class avg accuracy', str(np.mean(scores["nn"]["dist_accuracies"])))
    print('class avg accuracy', np.mean(scores["nn"]["dist_accuracies"]))
    mse = (scores["nn"]["mse"])
    mae = (scores["nn"]["mae"])
    mre = (scores["nn"]["mre"])
    writer.add_text('MSE', str(mse))
    print('MSE', str(mse))
    writer.add_text('MAE', str(mae))
    print('MAE', str(mae))
    writer.add_text('MRE', str(mre))
    print('MRE', str(mre))
    metrics = {"acc": np.mean(scores["nn"]["dist_accuracies"]), "mse": mse, "mae": mae,
               "mre": mre}
    return model

    """
        def dump_scores(trainer_scores):
                        if not graph_name in scores:
                            scores[graph_name] = {}
                        if not emb_dim in scores[graph_name]:
                            scores[graph_name][emb_dim] = {}
                        scores[graph_name][emb_dim][split] = trainer_scores
        dump_scores(scores)
    """

    """
    fig = plt.figure(figsize=(10,7))
    plt.subplot(2,1,1)
    plt.bar(range(18), dist_accuracies)
    for index, value in enumerate(dist_accuracies):
    plt.text(index+0.03, value, str(np.round(value, 2))+'%')
    plt.title('distance-wise accuracy')
    plt.xlabel('distance values')
    plt.ylabel('accuracy')
    plt.subplot(2,1,2)
    plt.bar(range(18), dist_counts)
    for index, value in enumerate(dist_counts):
    plt.text(index+0.03, value, str(value))
    plt.title('distance-wise count')
    plt.xlabel('distance values')
    plt.ylabel('counts')
    fig.tight_layout(pad=3.0)
    im_path = os.path.join(self.path, "accuracy scores.png")
    self.ensure_folders_exist(im_path)
    fig.savefig(im_path)
    writer.add_figure('test/results', fig)
    """
