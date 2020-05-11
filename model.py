from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlps import SplitLinear
from lstms import SplitLSTM

class SubModel(nn.Module):
    def __init__(self, num_const, num_var, num_hidden, num_out, num_groups):
        super(SubModel, self).__init__()
        self.lstm = SplitLSTM(num_const, num_var, num_hidden, num_groups, independent=True) # TODO make independent default
        self.fc = SplitLinear(0, num_hidden, num_out, num_groups) # TODO add independent parameter

    def forward(self, items, day=torch.tensor([])):
        lstm_out, hidden = self.lstm(items, day)
        lstm_out = lstm_out[:, -1] # take last day from sequence
        y = self.fc(lstm_out)
        return y

class Model(nn.Module):
    def __init__(self, num_const, num_var, num_hidden, num_out, num_groups, num_mlps):
        super(Model, self).__init__()

if __name__ == "__main__":
    device = torch.device('cpu')
    num_const = 12  # number of inputs per sub-LSTM that are constant per store-item
    num_var = 3  # number of inputs per sub-LSTM that are different per store-item
    horizon = 7  # number of hidden units per sub-LSTM and output of the entire model (= forecasting horizon)
    num_groups = 6  # number of store-item groups to make one LSTM for
    seq_len = 8  # sequence length, number of time points per forward pass

    model = SubModel(num_const, num_var, horizon, horizon, num_groups)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    day = torch.randn(1, seq_len, num_const).to(device)
    items = torch.randn(1, seq_len, num_groups, num_var).to(device)
    targets = torch.randn(1, num_groups, horizon).to(device)

    time = datetime.now()
    for _ in range(2**10):
        output = model(items, day)

        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(datetime.now() - time)
