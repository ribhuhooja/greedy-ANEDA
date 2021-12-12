import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
from typing import Union
from CustomDataset import CustomDataset
from data_helper import *
from evaluations import evaluate_metrics
from utils import is_numpy_array


class Trainer:
    @staticmethod
    def _make_train_step(model, loss_fn, optimizer):
        def train_step(x, y):
            model.train()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item()

        return train_step

    @staticmethod
    def _train(model, device, loss_fn, optimizer, lr_scheduler, n_epochs, train_loader, val_loader, verbose):
        train_step = Trainer._make_train_step(model, loss_fn, optimizer)

        training_losses = []
        validation_losses = []

        val_metrics_list = []

        for epoch in range(n_epochs):

            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss = train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            training_losses.append(training_loss)

            validation_loss, val_metrics = Trainer.evaluate_model(model, loss_fn, val_loader, device, evaluate_metrics)
            val_metrics_list.append(val_metrics)
            validation_losses.append(validation_loss)
            if verbose:
                print(
                    f"[epoch {epoch + 1}/{n_epochs}] Training loss: {training_loss:.4f};\tValidation loss: {validation_loss:.4f}, Validation metrics: {val_metrics}")

            if lr_scheduler:  # If using learning rate scheduler
                lr_scheduler.step()  # update learning rate after each epoch
        print("Done Training.")

        return val_metrics_list

    @staticmethod
    def evaluate_model(model, loss_fn, val_loader, device, evaluate_function=None):
        yhat_list = []
        ytrue_list = []
        val_losses = []
        metric_scores = None
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                model.eval()
                yhat = model(x_val)

                val_loss = loss_fn(yhat, y_val).item()
                val_losses.append(val_loss)

                if evaluate_function:
                    if device != "cpu":
                        yhat_list.append(yhat.cpu().numpy())
                        ytrue_list.append(y_val.cpu().numpy())
                    else:
                        yhat_list.append(yhat.numpy())
                        ytrue_list.append(y_val.numpy())

            validation_loss = np.mean(val_losses)
        if evaluate_function:
            metric_scores = evaluate_function(np.vstack(ytrue_list).reshape(-1), np.vstack(yhat_list).reshape(-1),
                                              print_out=False)
        return validation_loss, metric_scores

    @staticmethod
    def predict(model: nn.Module, x: Union[np.array, torch.Tensor]) -> np.array:
        """
        return predictions
        :param model:  neural net model
        :param x:  embedding of 2 nodes. x should be the average embedding of 2 nodes. See `data_helper.py` -> `create_dataset()` for more detail.
        :return: distance of the 2 nodes
        """
        if is_numpy_array(x):
            x = torch.tensor(x)

        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            yhat = model(x.to(device))

        if device != "cpu":
            return yhat.reshape(-1).cpu().numpy()
        else:
            return yhat.reshape(-1).numpy()

    @staticmethod
    def save_model(model, path, model_name):
        model_path = path.format(model_name=model_name)
        folder_path = os.path.dirname(path)  # create an output folder if it doesn't exist
        if not os.path.exists(folder_path):  # mkdir the folder to store output files
            os.makedirs(folder_path)

        torch.save(model.state_dict(), model_path)
        print(f"Model was saved at {model_path}")

    @staticmethod
    def load_model(model_class, params, model_name):
        try:
            model = model_class(params)
            model_path = params["save_path"].format(model_name=model_name)
            model.load_state_dict(torch.load(model_path))
            print(f"model was loaded from {model_path}")
        except:
            return None
        return model

    @staticmethod
    def compare_2_models(model_1, model_2):
        """
        return True if 2 models are the same, False otherwise
        :param model_1:
        :param model_2:
        :return:
        """
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if not torch.equal(key_item_1[1], key_item_2[1]):
                return False
        return True

    @staticmethod
    def train_model(model_class: nn.Module, dataset: Dict, params: Dict, test_dataset: Dict = None,
                    model_name: str = None):
        """
        Train a neural net
        :param model_class: Neural Net Class
        :param dataset: a dictionary contains keys=["x_train", "x_val", "y_train", "y_val"] and values are the corresponding datasets
        :param params: a dictionary contains params for the neural net
        :param test_dataset: [optional] a dictionary contains keys=["x_test", "y_test"] and values are the corresponding datasets
        :param model_name: model name to write to file. If None, not save the model
        :return: model, val metrics, and test metrics
        """
        # TODO: lr_finder, early_stopping

        train_dataset = CustomDataset(dataset["x_train"], dataset["y_train"])
        val_dataset = CustomDataset(dataset["x_val"], dataset["y_val"])

        train_loader = DataLoader(dataset=train_dataset, batch_size=params["batch_size"])
        val_loader = DataLoader(dataset=val_dataset, batch_size=params["batch_size"])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = model_class(params).to(device)

        optimizer = torch.optim.RMSprop(model.parameters(), lr=params['lr'], alpha=0.99, eps=1e-08, weight_decay=0,
                                        momentum=0, centered=False)
        loss_fn = nn.PoissonNLLLoss(log_input=False, eps=1e-07, reduction='mean')

        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, params['min_lr'], params['max_lr'],
                                                         step_size_up=8 * len(train_dataset), step_size_down=None,
                                                         mode=params['lr_sched_mode'], last_epoch=-1,
                                                         gamma=params['gamma'])
        summary(model, input_size=(params['input_size'],))  ## print out the model

        # Train the model
        val_metrics_list = Trainer._train(model, device, loss_fn, optimizer, lr_scheduler, params["epochs"],
                                          train_loader,
                                          val_loader, True)
        test_metrics = None
        if test_dataset:
            test_dataset = CustomDataset(test_dataset["x_test"], test_dataset["y_test"])
            test_loader = DataLoader(dataset=test_dataset, batch_size=params["batch_size"])
            test_metrics = Trainer.evaluate_model(model, loss_fn, test_loader, device,
                                                  evaluate_function=evaluate_metrics)

        if model_name is not None:
            Trainer.save_model(model, params["save_path"], model_name)

        return model, val_metrics_list, test_metrics  # return the model, and all the metrics on val, test sets

    @staticmethod
    def maybe_train_model(model_class: nn.Module, dataset: Dict, params: Dict, test_dataset: Dict = None,
                          model_name: str = None):
        """
        try to load model first, if not succeed, train the model
        :param model_class:
        :param dataset:
        :param params:
        :param test_dataset:
        :param model_name:
        :return: model, val_metrics_list, test_metrics
        """
        print("Try to load the model from disk first...")
        model = Trainer.load_model(model_class, params, model_name)

        if model is not None:  ## If we already had the model
            return model, None, None
        else:
            model, val_metrics_list, test_metrics = Trainer.train_model(model_class, dataset, params, test_dataset,
                                                                        model_name)
            return model, val_metrics_list, test_metrics
