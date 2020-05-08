from datetime import datetime

import torch
import torch.nn as nn


class SplitLSTM(nn.Module):
    """Module for LSTM with independent sub-layers."""

    def __init__(self, num_const, num_var, num_hidden, num_groups, independent):
        super(SplitLSTM, self).__init__()
        self.input_size = num_const + num_var * num_groups
        self.hidden_size = num_hidden * num_groups
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True)

        if independent:
            indices = self._weight_indices(num_const, num_var, num_hidden)
            self.row_indices, self.ih_col_indices, self.hh_col_indices = indices

            with torch.no_grad():
                self.lstm.weight_ih_l0[self.row_indices, self.ih_col_indices] = 0
                self.lstm.weight_ih_l0.register_hook(self._ih_split_hook)

                self.lstm.weight_hh_l0[self.row_indices, self.hh_col_indices] = 0
                self.lstm.weight_hh_l0.register_hook(self._hh_split_hook)

    def _weight_indices(self, num_const, num_var, num_hidden):
        row_indices = torch.arange(self.hidden_size * 4).view(-1, 1)

        ih_col_indices = []
        for i in range(num_const, self.input_size, num_var):
            ih_col_index = torch.cat((
                torch.arange(num_const, i),
                torch.arange(i + num_var, self.input_size)
            ))
            ih_col_indices.append(ih_col_index)
        ih_col_indices = torch.stack(ih_col_indices)
        ih_col_indices = ih_col_indices.repeat_interleave(num_hidden, dim=0)
        ih_col_indices = ih_col_indices.repeat(4, 1)

        hh_col_indices = []
        for i in range(0, self.hidden_size, num_hidden):
            hh_col_index = torch.cat((
                torch.arange(0, i),
                torch.arange(i + num_hidden, self.hidden_size)
            ))
            hh_col_indices.append(hh_col_index)
        hh_col_indices = torch.stack(hh_col_indices)
        hh_col_indices = hh_col_indices.repeat_interleave(num_hidden, dim=0)
        hh_col_indices = hh_col_indices.repeat(4, 1)

        return row_indices, ih_col_indices, hh_col_indices

    def _ih_split_hook(self, grad):
        grad = grad.clone()
        grad[self.row_indices, self.ih_col_indices] = 0

        return grad

    def _hh_split_hook(self, grad):
        grad = grad.clone()
        grad[self.row_indices, self.hh_col_indices] = 0

        return grad

    def forward(self, items, day=torch.tensor([]), hx=None):
        """Forward pass of split LSTM.

        Args:
            items = [torch.Tensor] data that is different per store-item
                of shape (batch_size, seq_len, num_groups, num_var)
            day   = [torch.Tensor] data that is constant per store-item
                of shape (batch_size, seq_len, num_const)
            hx    = [[torch.Tensor]*2] initial hidden and cell states
                of shape (1, batch_size, num_groups * num_hidden)

        Returns [[torch.Tensor, [torch.Tensor]*2]]:
            - Output of shape (batch_size, seq_len, num_groups, num_hidden).
            - * Hidden state of shape (1, batch_size, num_groups * num_hidden)
              * Cell state of shape (1, batch_size, num_groups * num_hidden)
        """
        input = torch.cat((day, items.flatten(start_dim=-2)), dim=-1)

        output, hidden = self.lstm(input, hx)        
        output = output.view(items.shape[:-1] + (-1,))

        return output, hidden


device = torch.device('cpu')
num_const = 12  # number of inputs per sub-LSTM that are constant per store-item
num_var = 3  # number of inputs per sub-LSTM that are different per store-item
horizon = 3  # number of hidden units per sub-LSTM
num_groups = 3  # number of store-item groups to make one LSTM for
seq_len = 8  # sequence length, number of time points per forward pass

lstm = SplitLSTM(num_const, num_var, horizon, num_groups, independent=True)
lstm.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters())

day = torch.randn(1, seq_len, num_const).to(device)
items = torch.randn(1, seq_len, num_groups, num_var).to(device)
targets = torch.randn(1, num_groups, horizon).to(device)

time = datetime.now()
for _ in range(2**10):
    output, hidden = lstm(items, day)
    output = output[:, 0]

    loss = criterion(output, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(datetime.now() - time)

# slower when dependent, but not a big difference
# 4.0s vs. 3.8s on CPU
# 6.4s vs. 5.8s on GPU

i = 3
