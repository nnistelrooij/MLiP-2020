from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_data(path):
    sales = pd.read_csv(path / 'sales_train_validation.csv')
    sales = sales[sales.columns[1:6].union(sales.columns[-28:])]
    sales = pd.wide_to_long(sales, 'd_', i=['store_id', 'item_id'], j='d')
    sales = sales.reset_index()
    sales['d'] = sales['d'].map(lambda x: f'd_{x}')

    cal = pd.read_csv(path / 'calendar.csv')
    cal = cal[['wm_yr_wk', 'd']]

    prices = pd.read_csv(path / 'sell_prices.csv')

    data = sales.merge(cal)
    data = data.merge(prices)
    data = data.sort_values(by=['store_id', 'item_id', 'd'])
    data.index = range(data.shape[0])
    data['revenue'] = data['d_'] * data['sell_price']
    data['total'] = 'TOTAL'

    return data


def level_indices(df):
    permutations = []
    group_indices = []

    groups1 = ['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id', 'total']
    groups2 = [['', ' cat_id', ' dept_id', ' item_id']]*2 + [['']]*4
    for group1, group2 in zip(groups1, groups2):
        for group2 in group2:
            groups_columns = (group1 + group2).split()
            groups = df.groupby(groups_columns)

            permutation = list(groups.indices.values())
            permutation.sort(key=lambda x: x[0])

            group_sizes = [0] + [len(group) for group in permutation]
            group_start_indices = np.cumsum(group_sizes)[:-1]
            group_indices.append(group_start_indices)

            permutation = np.concatenate(permutation)
            permutations.append(permutation)

    return permutations, group_indices


def aggregate(revenues, permutations, group_indices):
    revenues = revenues['revenue'].to_numpy()
    aggregates = []
    for permutation, group_start_indices in zip(permutations, group_indices):
        permutation = revenues[permutation]
        sums = np.add.reduceat(permutation, group_start_indices)

        aggregates.append(sums)

    return aggregates


def level_weights(aggregates):
    total = aggregates[-1]
    weights = []
    for level in aggregates:
        weights.append(level / total)

    return np.concatenate(weights)


def loss_weights(path):
    data = load_data(Path(path))
    permutations, group_indices = level_indices(data)
    aggregates = aggregate(data, permutations, group_indices)
    weights = level_weights(aggregates)

    return torch.tensor(weights)


if __name__ == '__main__':
    weights = loss_weights(
        r'D:\Users\Niels-laptop\Documents\2019-2020\Machine Learning in '
        r'Practice\Competition 2\project'
    )
