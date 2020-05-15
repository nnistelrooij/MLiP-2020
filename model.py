import math
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlps import SplitLinear
from lstms import SplitLSTM

class SubModel(nn.Module):
    def __init__(self, num_const, num_var, num_hidden, num_out, num_groups, num_batch):
        super(SubModel, self).__init__()
        self.num_hidden = num_hidden
        self.num_groups = num_groups
        self.num_batch = num_batch

        self.lstm = SplitLSTM(num_const, num_var, num_hidden, num_groups, independent=True) # TODO make independent default
        self.fc = SplitLinear(0, num_hidden, num_out, num_groups) # TODO add independent parameter for consistency

        self.hidden = (torch.zeros(1, num_batch, num_hidden*num_groups),
                       torch.zeros(1, num_batch, num_hidden*num_groups))
                    
    def reset_hidden(self):
        self.hidden = (torch.zeros(1, self.num_batch, self.num_hidden*self.num_groups),
                       torch.zeros(1, self.num_batch, self.num_hidden*self.num_groups))

    def forward(self, items, day=torch.tensor([])):
        lstm_out, self.hidden = self.lstm(items, day, self.hidden)
        lstm_out = lstm_out[:, -1] # take last day from sequence
        y = self.fc(lstm_out)
        return y

class Model(nn.Module):
    def __init__(self, num_const, num_var, num_hidden, num_out, num_groups, num_batch, num_submodels):
        super(Model, self).__init__()
        num_submodel_groups = math.floor(num_groups / num_submodels)
        num_extra_groups = num_groups % num_submodels
        self.num_groups = [num_submodel_groups + 1] * num_extra_groups
        self.num_groups += [num_submodel_groups] * (num_submodels - num_extra_groups)

        self.submodels = nn.ModuleList([SubModel(num_const, 
                                                 num_var, 
                                                 horizon, 
                                                 horizon, 
                                                 num_groups,
                                                 num_batch)
                                        for num_groups in self.num_groups])

    def reset_hidden(self):
        for submodel in submodels:
            submodel.reset_hidden()

    def forward(self, items, day):
        y = []
        for i, items in enumerate(items.split(self.num_groups, dim=-2)):
            y_part = self.submodels[i](items, day)
            y.append(y_part)

        return torch.cat(y, dim=-2)

if __name__ == "__main__":
    device = torch.device('cpu')
    num_const = 12  # number of inputs per sub-LSTM that are constant per store-item
    num_var = 3  # number of inputs per sub-LSTM that are different per store-item
    horizon = 5  # number of hidden units per sub-LSTM and output of the entire model (= forecasting horizon)
    num_groups = 3049 # number of store-item groups
    seq_len = 1  # sequence length, number of time points per forward pass
    num_batch = 1 # batch size (need to know batch size for LSTM hidden state initialization)

    model = Model(num_const, num_var, horizon, horizon, num_groups, num_batch, 100)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    day = torch.randn(1, seq_len, num_const).to(device)
    items = torch.randn(1, seq_len, num_groups, num_var).to(device)
    targets = torch.randn(1, num_groups, horizon).to(device)

    time = datetime.now()
    iterations = 2**6
    for _ in tqdm(range(iterations)):
        output = model(items, day)

        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    duration = datetime.now() - time
    print(f'Time for {iterations} iterations: ', duration)
    print(f'Time for one iterations: ', duration / iterations)
