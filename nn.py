import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class WRMSSE(nn.Module):
    """Weighted Root Mean Squared Scaled Error used for loss module."""
    _id_columns = ['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id']

    def __init__(self, device, calendar, prices, sales):
        """Initializes WRMSSE loss module.

        Args:
            device   = [torch.device] device to compute the loss on
            calendar = [pd.DataFrame] table with data on each date
            prices   = [pd.DataFrame] table with average sell prices each week
            sales    = [pd.DataFrame] table with sales per item for each day
        """
        super(WRMSSE, self).__init__()

        self._device = device

        sales = sales.sort_values(by=['store_id', 'item_id'])
        sales.index = range(sales.shape[0])
        self._permutations, self._group_indices = self._indices(sales)

        self._scales = self._time_series_scales(sales)
        self._weights = self._time_series_weights(calendar, prices, sales)

    def _time_series_scales(self, sales):
        """Computes the scale of each time series.

        Args:
            sales = [pd.DataFrame] table with sales per item for each day

        Returns [torch.Tensor]:
            Scale of each time series.
        """
        # select columns with unit sales and convert to torch.Tensor
        sales = torch.tensor(sales.filter(like='d_').to_numpy())

        # aggregate unit sales for each level of the time series hierarchy
        aggregates = self._aggregate(
            sales, self._permutations, self._group_indices
        )

        # compute scale of each time series
        squared_deltas = (aggregates[:, 1:] - aggregates[:, :-1])**2
        scales = torch.sum(squared_deltas, dim=1) / (sales.shape[1] - 1.0)

        return scales.to(self._device)

    def _time_series_weights(self, calendar, prices, sales):
        """Computes the weight of each time series.

        Args:
            calendar = [pd.DataFrame] table with data on each date
            prices   = [pd.DataFrame] table with average sell prices each week
            sales    = [pd.DataFrame] table with sales per item for each day
        """
        # select only necessary columns
        calendar = calendar[['wm_yr_wk', 'd']]

        # select only necessary columns and transform to long format data
        sales = sales[WRMSSE._id_columns], sales.filter(like='d_').iloc[:, -28:]
        sales = pd.concat(sales, axis=1)
        sales = pd.wide_to_long(sales, 'd_', i=['store_id', 'item_id'], j='d')
        sales = sales.reset_index()
        sales['d'] = sales['d'].map(lambda x: f'd_{x}')

        # create DataFrame with revenue data
        data = calendar.merge(sales)
        data = data.merge(prices)
        data = data.sort_values(by=['store_id', 'item_id', 'd'])
        data.index = range(data.shape[0])
        data['revenue'] = data['d_'] * data['sell_price']

        # determine group parameters from long format data
        permutations, group_indices = self._indices(data)

        # select column with revenues and convert to torch.Tensor
        revenues = torch.tensor(data['revenue'].to_numpy(), dtype=torch.float)

        # aggregate revenues for each level of the time series hierarchy
        aggregates = self._aggregate(revenues, permutations, group_indices)

        # compute weight of each time series
        total = aggregates[0]
        weights = aggregates / total

        return weights.to(self._device)

    @staticmethod
    def _indices(df):
        """Computes permutation and end indices of each group in input.

        Args:
            df = [pd.DataFrame] DataFrame with WRMSSE._id_columns columns

        Returns:
            permutations  = [[np.ndarray]*12] sales permutation for each group
            group_indices = [[np.ndarray]*12] end indices for each group
        """
        # add total column for highest level of hierarchy
        df['total'] = 'TOTAL'
        permutations = []
        group_indices = []

        col1 = ['total'] + WRMSSE._id_columns
        cols2 = [['']] + [[''] + WRMSSE._id_columns[-3:]]*2 + [['']]*3
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
            sums2 = torch.cat((torch.zeros_like(sales[:1]), sums1[:-1]))

            # add aggregate sum of sales to list
            aggregates.append(sums1 - sums2)

        return torch.cat(aggregates)

    def forward(self, input, target):
        """Computes the WRMSSE loss.

        Args:
            input  = [torch.Tensor] projected unit sales with shape (N, h)
            target = [torch.Tensor] actual unit sales with shape (N, h)

        Returns [torch.Tensor]:
            Tensor with a single value for the loss.
        """
        # aggregate the data to all levels of the time series hierarchy
        projected_sales = self._aggregate(
            input, self._permutations, self._group_indices
        )
        target = target.to(self._device)
        actual_sales = self._aggregate(
            target, self._permutations, self._group_indices
        )

        # compute WRMSSE loss
        horizon = input.shape[1]
        squared_errors = (actual_sales - projected_sales)**2
        MSE = torch.sum(squared_errors, dim=1) / horizon
        RMSSE = torch.sqrt(MSE / self._scales)
        loss = torch.sum(self._weights * RMSSE)

        return loss


if __name__ == '__main__':
    from datetime import datetime

    path = ('D:\\Users\\Niels-laptop\\Documents\\2019-2020\\Machine Learning in'
            ' Practice\\Competition 2\\project\\')
    calendar = pd.read_csv(path + 'calendar.csv')
    prices = pd.read_csv(path + 'sell_prices.csv')
    sales = pd.read_csv(path + 'sales_train_validation.csv')

    device = torch.device('cuda')
    time = datetime.now()
    criterion = WRMSSE(device, calendar, prices, sales)
    print('Time to initialize loss: ', datetime.now() - time)

    horizon = 5
    input = torch.rand(30490, horizon, device=device)
    target = torch.rand(30490, horizon)

    time = datetime.now()
    loss = criterion(input, target)
    print('Time to compute loss: ', datetime.now() - time)
