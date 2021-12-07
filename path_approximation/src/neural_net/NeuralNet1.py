from torch import nn
import torch


class NeuralNet1(nn.Module):
    '''
        A Simple Neural Net Model

    '''

    def __init__(self, params):
        super().__init__()
        self.layers = torch.nn.Sequential(
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
            torch.nn.Softplus()
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


