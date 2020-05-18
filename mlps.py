from datetime import datetime
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitLinear(nn.Linear):
    """Module for linear layer with independent sub-layers."""

    def __init__(self, num_const, num_var, num_out, num_groups, independent):
        """Initializes split linear layer.

        Args:
            num_const  = [int] number of input features constant per store-item
            num_var    = [int] number of input features different per store-item
            num_out    = [int] number of output features
            num_groups = [int] number of store-item groups to make models for
        """
        super(SplitLinear, self).__init__(
            in_features=num_const + num_var * num_groups,
            out_features=num_out * num_groups
        )

        if independent:
            indices = self._weight_indices(num_const, num_var, num_out)
            self.row_indices, self.col_indices = indices

            with torch.no_grad():
                self.weight[self.row_indices, self.col_indices] = 0
                self.weight.register_hook(self._split_hook)

    def _weight_indices(self, num_const, num_var, num_out):
        row_indices = torch.arange(self.out_features).view(-1, 1)

        col_indices = []
        for i in range(num_const, self.in_features, num_var):
            col_index = torch.cat((
                torch.arange(num_const, i),
                torch.arange(i + num_var, self.in_features)
            ))
            col_indices.append(col_index)
        col_indices = torch.stack(col_indices)
        col_indices = col_indices.repeat_interleave(num_out, dim=0)

        return row_indices, col_indices

    def _split_hook(self, grad):
        grad = grad.clone()
        grad[self.row_indices, self.col_indices] = 0

        return grad

    def forward(self, items, day=torch.tensor([])):
        """Forward pass of split linear layer.

        Args:
            items = [torch.Tensor] data that is different per store-item
                of shape (*, num_groups, num_var)
            day   = [torch.Tensor] data that is constant per store-item
                of shape (*, num_const)

        Returns:
            Output of shape (*, num_groups, num_out).
        """
        day = day.to(torch.device('cpu')) # TODO make device a member of SplitLinear
        x = torch.cat((day, items.flatten(start_dim=-2)), dim=-1)

        y = F.linear(x, self.weight, self.bias)
        return y.view(items.shape[:-1] + (-1,))


class SplitMLP(nn.Module):
    """Module that splits MLP with weight manipulation."""

    def __init__(self, num_const, num_var, num_hidden, num_out, num_groups):
        super(SplitMLP, self).__init__()

        self.fc1 = SplitLinear(num_const, num_var, num_hidden, num_groups, independent=True)
        self.fc2 = SplitLinear(0, num_hidden, num_out, num_groups, independent=True)

    def forward(self, day, items):
        """Forward pass of split MLP.

        Args:
            day   = [torch.Tensor] data that is constant per store-item
                of shape (*, num_const)
            items = [torch.Tensor] data that is different per store-item
                of shape (*, num_groups, num_var)

        Returns:
            Output of shape (*, num_groups, num_out).
        """
        h = F.relu(self.fc1(items, day))
        y = self.fc2(h)

        return y


class SplitModel(nn.Module):

    def __init__(self, num_const, num_var, num_hidden, num_out, num_groups, num_mlps):
        super(SplitModel, self).__init__()

        num_mlp_groups = math.floor(num_groups / num_mlps)
        num_extra_groups = num_groups % num_mlps
        self.num_groups = [num_mlp_groups + 1]*num_extra_groups
        self.num_groups += [num_mlp_groups]*(num_mlps - num_extra_groups)

        self.mlps = nn.ModuleList()

        for num_groups in self.num_groups:
            mlp = SplitMLP(num_const, num_var, num_hidden, num_out, num_groups)
            self.mlps.append(mlp)

    def forward(self, day, items):
        y = []
        for i, items in enumerate(items.split(self.num_groups, dim=-2)):
            y_part = self.mlps[i](day, items)
            y.append(y_part)

        return torch.cat(y, dim=-2)


if __name__ == '__main__':
    device = torch.device('cpu')
    num_const = 12
    num_var = 3
    horizon = 5
    num_groups = 3049

    model = SplitModel(num_const, num_var, 12, horizon, num_groups, 100)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    day = torch.randn(1, num_const).to(device)
    items = torch.randn(1, num_groups, num_var).to(device)
    targets = torch.randn(1, num_groups, horizon).to(device)

    time = datetime.now()
    iterations = 2**6
    for _ in range(iterations):
        iter_time = datetime.now()
        y = model(day, items)

        loss = criterion(y, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('Time for one iteration: ', datetime.now() - iter_time)
    print(f'Time for {iterations}iterations: ', datetime.now() - time)

    i = 3

