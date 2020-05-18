import math
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlps import SplitLinear
from lstms import SplitLSTM

class SubModel(nn.Module):
    def __init__(self, num_const, num_var, num_hidden, num_out, num_groups, independent):
        super(SubModel, self).__init__()
        self.lstm = SplitLSTM(num_const, num_var, num_hidden, num_groups, independent)
        self.bn = nn.BatchNorm1d(num_groups)
        self.fc = SplitLinear(0, num_hidden, num_out, num_groups, independent)
                    
    def reset_hidden(self):
        self.lstm.reset_hidden()

    def forward(self, day, items):
        lstm_out, hidden = self.lstm(items, day)
        lstm_out = lstm_out[:, -1] # take last day from sequence
        y = self.fc(lstm_out)
        return y

class Model(nn.Module):
    def __init__(self, num_const, num_var, num_hidden, num_out, num_groups, num_submodels, device, independent=True):
        super(Model, self).__init__()
        num_submodel_groups = math.floor(num_groups / num_submodels)
        num_extra_groups = num_groups % num_submodels
        self.num_groups = [num_submodel_groups + 1] * num_extra_groups
        self.num_groups += [num_submodel_groups] * (num_submodels - num_extra_groups)

        self.submodels = nn.ModuleList([SubModel(num_const, 
                                                 num_var, 
                                                 num_hidden,
                                                 num_out,
                                                 num_groups,
                                                independent)
                                        for num_groups in self.num_groups]

        self.device = device
                                       
    def reset_hidden(self):
        for submodel in submodels:
            submodel.reset_hidden()     

    def forward(self, day, items):
        day = day.to(self.device)
        items = items.to(self.device)
                                       
        y = []
        for i, items in enumerate(items.split(self.num_groups, dim=-1)):
            y_part = self.submodels[i](day, items)
            y.append(y_part)

        return torch.cat(y, dim=-2)

if __name__ == "__main__":
    device = torch.device('cpu')
    num_const = 12  # number of inputs per sub-LSTM that are constant per store-item
    num_var = 3  # number of inputs per sub-LSTM that are different per store-item
    horizon = 5  # number of hidden units per sub-LSTM and output of the entire model (= forecasting horizon)
    num_groups = 3049 # number of store-item groups
    seq_len = 1  # sequence length, number of time points per forward pass

    model = Model(num_const, num_var, horizon, horizon, num_groups, 100)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    day = torch.randn(1, seq_len, num_const).to(device)
    items = torch.randn(1, seq_len, num_groups, num_var).to(device)
    targets = torch.randn(1, num_groups, horizon).to(device)

    time = datetime.now()
    iterations = 2**6
    for _ in tqdm(range(iterations)):
        output = model(day, items)

        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    duration = datetime.now() - time
    print(f'Time for {iterations} iterations: ', duration)
    print(f'Time for one iterations: ', duration / iterations)
