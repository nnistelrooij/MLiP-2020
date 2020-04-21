from pathlib import Path

import pandas as pd
import numpy as np
import torch

from weights import level_indices


def load_sales(path):
    sales = pd.read_csv(path / 'sales_train_validation.csv')
    sales = sales.iloc[:, 1:]
    sales = sales.sort_values(by=['store_id', 'item_id'])
    sales.index = range(sales.shape[0])
    sales['total'] = 'TOTAL'

    return sales


def aggregate(sales, permutations, group_indices):
    sales = sales[[f'd_{i}' for i in range(1, 1914)]].to_numpy()
    aggregates = []
    for permutation, group_start_indices in zip(permutations, group_indices):
        permutation = sales[permutation]
        sums = np.add.reduceat(permutation, group_start_indices)

        aggregates.append(sums)

    return np.concatenate(aggregates)


def level_scales(aggregates):
    squared_deltas = (aggregates[:, 1:] - aggregates[:, :-1])**2
    scales = np.sum(squared_deltas, axis=1) / squared_deltas.shape[1]

    return scales


def loss_scales(path):
    sales = load_sales(Path(path))
    permutations, group_indices = level_indices(sales)
    aggregates = aggregate(sales, permutations, group_indices)
    scales = level_scales(aggregates)

    return torch.tensor(scales)


if __name__ == '__main__':
    scales = loss_scales(
        r'D:\Users\Niels-laptop\Documents\2019-2020\Machine Learning in '
        r'Practice\Competition 2\project'
    )
