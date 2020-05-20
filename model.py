from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torchsummary import summary

from mlps import SplitLinear
from lstms import SplitLSTM

num_const = 32
num_var = 3
num_hidden = 5
num_groups = 30490


class SubModel(nn.Module):
    """Class that implements LSTM network for subset of all store-item groups.

    Attributes:
        lstm = [SplitLSTM] LSTM part of the submodel
        fc   = [SplitLinear] fully-connected part of the submodel
    """
    def __init__(self, num_out, num_groups, independent):
        """Initializes the submodel.

        Args:
            num_const   = [int] number of inputs constant per store-item group
            num_var     = [int] number of inputs different per store-item group
            num_hidden  = [int] number of hidden units per store-item group
            num_out     = [int] number of output units per store-item group
            num_groups  = [int] number of store-item groups to make a model for
            independent = [bool] whether the submodel has independent groups
        """
        super(SubModel, self).__init__()

        self.lstm = SplitLSTM(
            num_const, num_var, num_hidden, num_groups, independent
        )
        self.fc = SplitLinear(num_hidden, num_out, num_groups, independent)
                    
    def reset_hidden(self):
        """Resets the hidden state of the LSTM."""
        self.lstm.reset_hidden()

    def forward(self, day, items):
        lstm_out, hidden = self.lstm(items, day)
        lstm_out = lstm_out[:, -1] # take last day from sequence
        y = self.fc(lstm_out)
        return y


class Model(nn.Module):
    """Class that implements LSTM network for all store-item groups.

    Attributes:
        num_const        = [int] number of inputs constant per store-item group
        num_var          = [int] number of input different per store-item group
        num_hidden       = [int] number of hidden units per store-item group
        num_out          = [int] number of output units per store-item group
        num_groups       = [int] number of store-item groups to make models for
        num_models       = [int] number of models to make
        num_model_groups = [[int]*num_models] number of groups of each model
        submodels        = [nn.ModuleList] list of submodels
        device           = [torch.device] device to put the model and data on
    """

    def __init__(self, num_out, num_models, device, independent=True):
        """Initializes the model.

        Args:
            num_out     = [int] number of output units per store-item group
            num_models  = [int] number of models to make
            device      = [torch.device] device to put the model and data on
            independent = [bool] whether each submodel has independent groups
        """
        super(Model, self).__init__()

        min_model_groups = self.num_groups // num_models
        num_extra_groups = self.num_groups % num_models
        self.num_model_groups = torch.full([self.num_models], min_model_groups)
        self.num_model_groups[:num_extra_groups] = min_model_groups + 1
        self.num_model_groups = self.num_model_groups.tolist()

        self.submodels = nn.ModuleList()
        for num_groups in Model.num_model_groups:
            submodel = SubModel(num_out, num_groups, independent)
            submodel = submodel.to(device)
            self.submodels.append(submodel)

        self.device = device

    def reset_hidden(self):
        """Resets the hidden states of all submodels."""
        for submodel in self.submodels:
            submodel.reset_hidden()



    def forward(self, day, items):
        """Forward pass of the model.

        `items` is split, such that each submodel receives the correct number
        of inputs. Each submodel is run in order without concurrency.

        Args:
            day   = [torch.Tensor] inputs constant per store-item group
                The shape should be (1, seq_len, Model.num_const)
            items = [torch.Tensor] inputs different per store-item group
        """
        day = day.to(self.device)
        items = items.to(self.device)

        y = []
        for i, items in enumerate(items.split(self.num_model_groups, dim=-1)):
            submodel = self.submodels[i]
            y.append(submodel(day, items))

        return torch.cat(y, dim=-2)

# if __name__ == "__main__":
    # device = torch.device('cpu')
    # num_const = 12  # number of inputs per sub-LSTM that are constant per store-item
    # num_var = 3  # number of inputs per sub-LSTM that are different per store-item
    # horizon = 5  # number of hidden units per sub-LSTM and output of the entire model (= forecasting horizon)
    # num_groups = 3049 # number of store-item groups
    # seq_len = 1  # sequence length, number of time points per forward pass
    #
    # model = Model(num_const, num_var, horizon, horizon, num_groups, 100)
    #
    # model.to(device)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters())
    #
    # day = torch.randn(1, seq_len, num_const).to(device)
    # items = torch.randn(1, seq_len, num_groups, num_var).to(device)
    # targets = torch.randn(1, num_groups, horizon).to(device)
    #
    # time = datetime.now()
    # iterations = 2**6
    # for _ in tqdm(range(iterations)):
    #     output = model(day, items)
    #
    #     loss = criterion(output, targets)
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    # duration = datetime.now() - time
    # print(f'Time for {iterations} iterations: ', duration)
    # print(f'Time for one iterations: ', duration / iterations)
