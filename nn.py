import math

import numpy as np
import pandas as pd
import torch
from torch.distributions.bernoulli import Bernoulli
import torch.nn as nn
import torch.nn.functional as F

from utils.data import ForecastDataset

num_const = 20
num_hidden = 5


class WRMSSE(nn.Module):
    """Weighted Root Mean Squared Scaled Error as loss module.

    Attributes:
        device        = [torch.device] device to compute loss on
        permutations  = [[np.ndarray]*12] sales permutations for each level
        group_indices = [[np.ndarray]*12] end indices of each group and level
        scales        = [torch.Tensor] pre-computed scales used in the WRMSSE
        weights       = [torch.Tensor] pre-computed weights used in the WRMSSE
    """

    def __init__(self, calendar, prices, sales, device):
        """Initializes WRMSSE loss module.

        Args:
            calendar = [pd.DataFrame] table with data on each date
            prices   = [pd.DataFrame] table with average sell prices each week
            sales    = [pd.DataFrame] table with sales per item for each day
            device   = [torch.device] device to compute the loss on
        """
        super(WRMSSE, self).__init__()

        self.device = device

        self.permutations, self.group_indices = self._indices(sales)

        self.scales = self._time_series_scales(sales)
        self.weights = self._time_series_weights(calendar, prices, sales)

    def _time_series_scales(self, sales):
        """Computes the scale of each time series.

        Args:
            sales = [pd.DataFrame] table with sales per item for each day

        Returns [torch.Tensor]:
            Scale of each time series.
        """
        # select columns with unit sales and convert to torch.Tensor
        sales = torch.tensor(sales.filter(like='d_').to_numpy()).float()

        # aggregate unit sales for each level of the time series hierarchy
        aggregates = self._aggregate(
            sales, self.permutations, self.group_indices
        )

        # compute scale of each time series
        squared_deltas = (aggregates[:, 1:] - aggregates[:, :-1])**2
        scales = torch.sum(squared_deltas, dim=1) / (sales.shape[1] - 1)

        return scales.to(self.device)

    def _time_series_weights(self, calendar, prices, sales):
        """Computes the weight of each time series.

        Args:
            calendar = [pd.DataFrame] table with data on each date
            prices   = [pd.DataFrame] table with average sell prices each week
            sales    = [pd.DataFrame] table with sales per item for each day

        Returns [torch.Tensor]:
            Weight of each time series.
        """
        # select only necessary columns
        calendar = calendar[['wm_yr_wk', 'd']]

        # select only necessary columns and transform to long format data
        sales = sales.filter(like='_id'), sales.filter(like='d_').iloc[:, -28:]
        sales = pd.concat(sales, axis=1)
        sales = pd.wide_to_long(sales, 'd_', i=['store_id', 'item_id'], j='d')
        sales = sales.reset_index()
        sales['d'] = sales['d'].map(lambda x: f'd_{x}')

        # create DataFrame with revenue data
        data = calendar.merge(sales)
        data = prices.merge(data)
        data['revenue'] = data['d_'] * data['sell_price']

        # determine group parameters from long format data
        permutations, group_indices = self._indices(data)

        # select column with revenues and convert to torch.Tensor
        revenues = torch.tensor(data['revenue'].to_numpy()).float()

        # aggregate revenues for each level of the time series hierarchy
        aggregates = self._aggregate(revenues, permutations, group_indices)

        # compute weight of each time series
        total = aggregates[0]
        weights = aggregates / total

        return weights.to(self.device)

    @staticmethod
    def _indices(df):
        """Computes permutation and end indices of each group in input.

        Args:
            df = [pd.DataFrame] DataFrame with columns like *_id

        Returns:
            permutations  = [[np.ndarray]*12] sales permutation for each level
            group_indices = [[np.ndarray]*12] end indices of each group & level
        """
        # add total column for highest level of hierarchy
        df['total'] = 'TOTAL'
        permutations = []
        group_indices = []

        col1 = ['total', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id']
        cols2 = [['']] + [['', 'cat_id', 'dept_id', 'item_id']]*2 + [['']]*3
        for column1, col2 in zip(col1, cols2):
            for column2 in col2:
                level_columns = f'{column1} {column2}'.split()
                groups = df.groupby(level_columns)

                permutation = groups.indices.values()
                permutation = sorted(permutation, key=lambda x: x[0])

                group_sizes = [len(group) for group in permutation]
                group_end_indices = np.cumsum(group_sizes) - 1
                group_indices.append(group_end_indices)

                permutation = np.concatenate(permutation)
                permutations.append(permutation)

        return permutations, group_indices

    @staticmethod
    def _aggregate(sales, permutations, group_indices):
        """Aggregates sales to compute all levels of the time series hierarchy.

        Args:
            sales         = [torch.Tensor] Tensor of unit or dollar sales
            permutations  = [[np.ndarray]*12] sales permutation for each group
            group_indices = [[np.ndarray]*12] end indices for each group

        Returns [torch.Tensor]:
            Tensor of aggregate sales given the permutations and group indices.
        """
        aggregates = []
        for permutation, group_end_indices in zip(permutations, group_indices):
            # permute input to get consecutive groups
            permutation = sales[permutation]

            # compute cumulative sum of sales along each group
            sums1 = permutation.cumsum(0)[group_end_indices]
            sums2 = torch.cat((torch.zeros_like(sums1[:1]), sums1[:-1]))

            # add aggregate sum of sales to list
            aggregates.append(sums1 - sums2)

        return torch.cat(aggregates)

    def forward(self, input, target):
        """Computes the WRMSSE loss.

        Args:
            input  = [torch.Tensor] projected sales of shape (horizon, 30490)
            target = [torch.Tensor] actual sales with shape (horizon, 30490)

        Returns [torch.Tensor]:
            Tensor with a single value for the loss.
        """
        # aggregate to all levels of the hierarchy
        projected_sales = self._aggregate(
            input.T, self.permutations, self.group_indices
        )

        # put target on GPU and aggregate to all levels of the hierarchy
        target = target.to(self.device)
        actual_sales = self._aggregate(
            target.T, self.permutations, self.group_indices
        )

        # compute WRMSSE loss
        squared_errors = (actual_sales - projected_sales)**2
        MSE = torch.sum(squared_errors, dim=1) / input.shape[0]
        RMSSE = torch.sqrt(MSE / self.scales + 1e-18)  # numerical instability
        loss = torch.sum(self.weights * RMSSE)

        return loss


class Dropout(nn.Module):
    """Module to drop out weights and gradients on specific indices.

    Attributes:
        q            = [float] prob of keeping inter-group weight or gradient
        weight_idx   = [[torch.Tensor]*2] index arrays to apply Dropout to
        sample_shape = [torch.Size] shape of the inter-group weights subset
        bernoulli    = [Bernoulli] Bernoulli sample distribution
        sample       = [torch.Tensor] previous sample of Bernoulli distribution
    """

    def __init__(self, p, weight_idx):
        """Initializes Dropout module.

        Args:
            p        = [torch.Tensor] prob to drop inter-group weight or grad
            weight_idx = [[torch.Tensor]*2] index arrays to apply Dropout to
        """
        super(Dropout, self).__init__()

        self.q = 1 - p + 1e-18
        self.weight_idx = weight_idx
        self.sample_shape = weight_idx[1].shape
        self.bernoulli = Bernoulli(1 - p)
        self.sample = None

    def forward(self, input, new_sample):
        """Forward pass of Dropout module.

        Args:
            input      = [torch.Tensor] weights or grads to apply Dropout to
            new_sample = [bool] whether to draw a new Bernoulli sample

        Returns [torch.Tensor]:
            Weights or gradients, where on some of the indices specified in
            self.weight_idx, they are set to zero during training.
        """
        # keep all weights when validating
        if not self.training:
            return input

        # one new sample per iteration
        if new_sample:
            self.sample = self.bernoulli.sample(self.sample_shape)

        # drop out inter-group weights or gradients
        input = input.clone()
        input[self.weight_idx] *= self.sample

        # adjust weights during training for consistency during validation
        if input.requires_grad:
            input[self.weight_idx] /= self.q

        return input


class Linear(nn.Module):
    """Module for fully-connected layer for multiple groups.

    The inter-group weights and gradients are dropped with probability dropout
    to control model complexity.

    Attributes:
        weight  = [nn.Parameter] parameter that holds the FC layer weights
        bias    = [nn.Parameter] parameter that holds the FC layer biases
        dropout = [Dropout] drops inter-group weights and gradients
    """

    def __init__(self, num_groups, dropout):
        """Initializes linear layer for multiple groups.

        Args:
            num_groups = [int] number of store-item groups to make layer for
            dropout    = [torch.Tensor] prob to drop inter-group weight or grad
        """
        global num_hidden

        super(Linear, self).__init__()

        # initialize parameters of fully-connected layer
        in_features = num_hidden * num_groups
        out_features = num_groups
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        # register Dropout backward hook
        weight_idx = self._weight_indices(in_features, out_features)
        self.dropout = Dropout(p=dropout, weight_idx=weight_idx)
        self.weight.register_hook(self._hook)

    def reset_parameters(self):
        """Reset parameters of fully-connected layer."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _weight_indices(in_features, out_features):
        """Compute index arrays of inter-group weights.

        Args:
            in_features  = [int] total number of inputs
            out_features = [int] total number of outputs

        Returns [[torch.Tensor]*2]:
            Index arrays for easily dropping inter-group weights and gradients.
        """
        global num_hidden

        row_indices = torch.arange(out_features).view(-1, 1)

        col_indices = []
        for i in range(0, in_features, num_hidden):
            col_index = torch.cat((
                torch.arange(0, i),
                torch.arange(i + num_hidden, in_features)
            ))
            col_indices.append(col_index)
        col_indices = torch.stack(col_indices)

        return row_indices, col_indices

    def _hook(self, grad):
        """Backward hook to drop inter-group weight gradients."""
        return self.dropout(grad, new_sample=False)

    def forward(self, input):
        """Forward pass of the linear layer.

        Args:
            input = [torch.Tensor] input of shape (T, num_groups * num_hidden)

        Returns:
            Output of shape (T, num_groups).
        """
        weight = self.dropout(self.weight, new_sample=True)

        return F.linear(input, weight, self.bias)


class LSTM(nn.RNNBase):
    """Module for LSTM for multiple groups.

    The inter-group weights and gradients are dropped with probability dropout
    to control model complexity.

    Attributes:
        hx         = [[torch.Tensor]*2] hidden and cell states of LSTM
        dropout_ih = [Dropout] drops input-hidden inter-group weights and grads
        dropout_hh = [Dropout] drops hidden-hidden inter-group weights and grads
    """

    def __init__(self, num_groups, dropout):
        """Initializes LSTM for multiple groups.

        Args:
            num_groups = [int] number of store-item groups to make LSTM for
            dropout    = [torch.Tensor] prob to drop inter-group weight or grad
        """
        global num_const
        global num_hidden

        # initialize LSTM and hidden state of LSTM
        input_size = num_const + ForecastDataset.num_var * num_groups
        hidden_size = num_hidden * num_groups
        super(LSTM, self).__init__(
            'LSTM', input_size, hidden_size, batch_first=True
        )
        self.hx = None

        # compute index arrays and initialize Dropout modules
        weight_indices = self._weight_indices(input_size, hidden_size)
        self.dropout_ih = Dropout(p=dropout, weight_idx=weight_indices[0])
        self.dropout_hh = Dropout(p=dropout, weight_idx=weight_indices[1])

        # register input-hidden and hidden-hidden backward hooks
        self.weight_ih_l0.register_hook(self._ih_hook)
        self.weight_hh_l0.register_hook(self._hh_hook)

    def check_forward_args(self, input, hidden, batch_sizes=None):
        """Check validity of shapes of input and hidden and cell states."""
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden[0], expected_hidden_size,
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], expected_hidden_size,
                               'Expected hidden[1] size {}, got {}')

    def reset_hidden(self):
        """Reset hidden state of LSTM."""
        self.hx = None

    @staticmethod
    def _weight_indices(input_size, hidden_size):
        """Compute index arrays of inter-group weights.

        Args:
            input_size  = [int] total number of inputs
            hidden_size = [int] total number of hidden units

        Returns [[[torch.Tensor]*2]*2]:
            Index arrays for easily dropping inter-group weights and gradients.
        """
        global num_const
        global num_hidden

        row_indices = torch.arange(hidden_size * 4).view(-1, 1)

        col_ih_indices = []
        for i in range(num_const, input_size, ForecastDataset.num_var):
            col_ih_index = torch.cat((
                torch.arange(num_const, i),
                torch.arange(i + ForecastDataset.num_var, input_size)
            ))
            col_ih_indices.append(col_ih_index)
        col_ih_indices = torch.stack(col_ih_indices)
        col_ih_indices = col_ih_indices.repeat_interleave(num_hidden, dim=0)
        col_ih_indices = col_ih_indices.repeat(4, 1)

        col_hh_indices = []
        for i in range(0, hidden_size, num_hidden):
            col_hh_index = torch.cat((
                torch.arange(0, i),
                torch.arange(i + num_hidden, hidden_size)
            ))
            col_hh_indices.append(col_hh_index)
        col_hh_indices = torch.stack(col_hh_indices)
        col_hh_indices = col_hh_indices.repeat_interleave(num_hidden, dim=0)
        col_hh_indices = col_hh_indices.repeat(4, 1)

        weight_ih_idx = row_indices, col_ih_indices
        weight_hh_idx = row_indices, col_hh_indices
        return weight_ih_idx, weight_hh_idx

    def _ih_hook(self, grad):
        """Backward hook to drop input-hidden inter-group weight gradients."""
        return self.dropout_ih(grad, new_sample=False)

    def _hh_hook(self, grad):
        """Backward hook to drop hidden-hidden inter-group weight gradients."""
        return self.dropout_hh(grad, new_sample=False)

    def _detach_hidden(self):
        """Detach hidden state from computational graph for next iteration."""
        h_n, c_n = self.hx
        self.hx = h_n.detach(), c_n.detach()

    def forward(self, input, keep_hidden):
        """Forward pass of the LSTM.

        If keep_hidden = True, the hidden and cell states are saved in self.hx
        without detaching them from the computational graph. Otherwise, the
        current self.hx is kept and detached from the computational graph.

        Args:
            input       = [torch.Tensor] input of shape
                (1, seq_len, num_const + num_groups * num_var)
            keep_hidden = [bool] keep hidden and cell states in self.hx

        Returns [torch.Tensor]:
            Output of shape (1, seq_len, num_groups * num_hidden).
        """
        # initialize hidden and cell states
        if self.hx is None:
            zeros = torch.zeros(1, 1, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            self.hx = zeros, zeros

        # set some inter-group weights to zero with Dropout
        weight_ih = self.dropout_ih(self.weight_ih_l0, new_sample=keep_hidden)
        weight_hh = self.dropout_hh(self.weight_hh_l0, new_sample=keep_hidden)
        flat_weights = [weight_ih, weight_hh, self.bias_ih_l0, self.bias_hh_l0]

        # run LSTM on input and parameters
        self.check_forward_args(input, self.hx)
        result = nn._VF.lstm(input, self.hx, flat_weights, self.bias,
                             self.num_layers, self.dropout, self.training,
                             self.bidirectional, self.batch_first)
        y, hidden = result[0], result[1:]

        # set or detach hidden and cell states for next iteration
        if keep_hidden:
            self.hx = hidden
        else:
            self._detach_hidden()

        return y


class SubModel(nn.Module):
    """Class that implements LSTM network for subset of all store-item groups.

    Attributes:
        lstm = [LSTM] LSTM part of the submodel
        fc   = [Linear] fully-connected part of the submodel
    """

    def __init__(self, num_groups, dropout):
        """Initializes the submodel.

        Args:
            num_groups = [int] number of store-item groups to make submodel for
            dropout    = [torch.Tensor] prob to drop inter-group weight gradient
        """
        super(SubModel, self).__init__()

        self.lstm = LSTM(num_groups, dropout)
        self.fc = Linear(num_groups, dropout)

    def reset_hidden(self):
        """Resets the hidden state of the LSTM."""
        self.lstm.reset_hidden()

    def forward(self, day, t_day, items, t_items):
        """Forward pass of the submodel.

        Args:
            day   = [torch.Tensor] inputs constant per store-item group
                The shape should be (1, seq_len, num_const).
            t_day   = [torch.Tensor] targets different per store-item group
                The shape should be (1, horizon, num_const).
            items = [torch.Tensor] inputs different per store-item group
                The shape should be (1, seq_len, num_groups, num_var).
            t_items = [torch.Tensor] targets different per store-item group
                The shape should be (1, horizon, num_groups, num_var).

        Returns [torch.Tensor]:
            Output of shape (horizon, num_groups).
        """
        # put inputs in one tensor
        x = torch.cat((day, items.flatten(start_dim=-2)), dim=-1)
        t_x = torch.cat((t_day, t_items.flatten(start_dim=-2)), dim=-1)

        # run LSTM on inputs
        self.lstm(x, keep_hidden=True)

        # run LSTM again on targets
        h = self.lstm(t_x, keep_hidden=False)

        # run linear layer on output of LSTM
        y = self.fc(h)

        return y.squeeze(0)


class Model(nn.Module):
    """Class that implements LSTM networks for all store-item groups.

    Attributes:
        num_model_groups = [[int]*num_models] number of groups of each submodel
        submodels        = [nn.ModuleList] list of submodels
        device           = [torch.device] device to put the model and data on
    """

    def __init__(self, num_models, dropout, device):
        """Initializes the model.

        Args:
            num_models = [int] number of submodels to make
            dropout    = [float] prob of dropping inter-group weight gradient
            device     = [torch.device] device to put the models and data on
        """
        global num_const

        super(Model, self).__init__()

        # make linear layer for embeddings constant per store-item group
        self.fc = nn.Linear(ForecastDataset.num_const, num_const).to(device)

        # determine number of groups each submodel will model
        min_model_groups = ForecastDataset.num_groups // num_models
        num_extra_groups = ForecastDataset.num_groups % num_models
        self.num_model_groups = torch.full((num_models,), min_model_groups)
        self.num_model_groups[:num_extra_groups] = min_model_groups + 1
        self.num_model_groups = self.num_model_groups.long().tolist()

        # initialize submodels as a ModuleList on the correct device
        dropout = torch.tensor(dropout).to(device)
        self.submodels = nn.ModuleList()
        for num_model_groups in self.num_model_groups:
            submodel = SubModel(num_model_groups, dropout).to(device)
            self.submodels.append(submodel)

        self.device = device

    def reset_hidden(self):
        """Resets the hidden states of all submodels."""
        for submodel in self.submodels:
            submodel.reset_hidden()

    def forward(self, day, t_day, items, t_items):
        """Forward pass of the model.

        items and t_items are split, such that each submodel receives the
        correct number of inputs and targets. Each submodel is run in order
        without making use of concurrency.

        Args:
            day   = [torch.Tensor] inputs constant per store-item group
                The shape should be (1, seq_len, num_const).
            t_day   = [torch.Tensor] targets different per store-item group
                The shape should be (1, horizon, num_const).
            items = [torch.Tensor] inputs different per store-item group
                The shape should be (1, seq_len, num_groups, num_var).
            t_items = [torch.Tensor] targets different per store-item group
                The shape should be (1, horizon, num_groups, num_var).

        Returns [torch.Tensor]:
            Output of shape (horizon, num_groups)
        """
        # put input on correct device and split for each submodel
        day = day.to(self.device)
        t_day = t_day.to(self.device)
        items = items.to(self.device).split(self.num_model_groups, dim=-2)
        t_items = t_items.to(self.device).split(self.num_model_groups, dim=-2)

        # get embeddings from data constant per store-item group
        day = F.relu(self.fc(day))
        t_day = F.relu(self.fc(t_day))

        y = []
        for submodel, items, t_items in zip(self.submodels, items, t_items):
            sub_y = submodel(day, t_day, items, t_items)
            y.append(sub_y)

        return torch.cat(y, dim=-1)


if __name__ == '__main__':
    from datetime import datetime

    # model = Model(1500, torch.device('cpu'))

    linear = Linear(100, 0.5)

    wolla = linear.dropout(torch.randn(100, 500))

    day = torch.randn(1, 6, 29)
    items = torch.randn(1, 6, 30490, 3)
    t_day = torch.randn(1, 3, 29)
    t_items = torch.randn(1, 3, 30490, 3)

    y = model(day, items, t_day, t_items)

    path = ('D:\\Users\\Niels-laptop\\Documents\\2019-2020\\Machine Learning '
            'in Practice\\Competition 2\\project\\')
    calendar = pd.read_csv(path + 'calendar.csv')
    prices = pd.read_csv(path + 'sell_prices.csv')
    sales = pd.read_csv(path + 'sales_train_validation.csv')

    # device = torch.device('cuda')
    device = torch.device('cpu')
    time = datetime.now()
    criterion = WRMSSE(device, calendar, prices, sales)
    print('Time to initialize loss: ', datetime.now() - time)

    horizon = 5
    input = torch.rand(30490, horizon, device=device)
    target = torch.rand(1, 30490, 3)

    time = datetime.now()
    loss = criterion(input, target)
    print('Time to compute loss: ', datetime.now() - time)
