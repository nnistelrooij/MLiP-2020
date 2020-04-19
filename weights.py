import numpy as np
import pandas as pd
import torch

sales = pd.read_csv(
    r'D:\Users\Niels-laptop\Documents\2019-2020\Machine Learning '
    r'in Practice\Competition 2\project\sales_train_validation.csv'
)
sales = sales.iloc[:, list(range(1, 6)) + list(range(sales.shape[1]))[-28:]]
sales = pd.wide_to_long(sales, stubnames='d_', i=['store_id', 'item_id'], j='d')
sales = sales.reset_index()
sales['d'] = sales['d'].map(lambda x: f'd_{x}')

cal = pd.read_csv(
    r'D:\Users\Niels-laptop\Documents\2019-2020\Machine Learning '
    r'in Practice\Competition 2\project\calendar.csv'
)
cal = cal[['wm_yr_wk', 'd']]

prices = pd.read_csv(
    r'D:\Users\Niels-laptop\Documents\2019-2020\Machine '
    r'Learning in Practice\Competition 2\project\sell_prices.csv'
)


def level_indices(df):
    indices = []

    groups1 = ['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id', 'total']
    groups2 = [['', ' cat_id', ' dept_id', ' item_id']]*2 + [['']]*4
    for group1, group2 in zip(groups1, groups2):
        for group2 in group2:
            groups_columns = (group1 + group2).split()
            groups = df.groupby(groups_columns)

            permutation = list(groups.indices.values())
            permutation.sort(key=lambda x: x[0])

            group_sizes = [len(group) for group in permutation]
            group_indices = np.cumsum(group_sizes) - 1

            permutation = np.concatenate(permutation)
            indices.append((permutation, group_indices))

    return indices


def aggregate(revenues, indices):
    revenues = torch.tensor(revenues['revenue'].to_numpy())
    aggregates = []
    for permutation, group_indices in indices:
        permutation = revenues[permutation]
        sums1 = permutation.cumsum(0)[group_indices]
        sums2 = torch.cat((torch.zeros_like(revenues[:1]), sums1[:-1]))

        aggregates.append(sums1 - sums2)

    return aggregates


def level_weights(aggregates):
    total = aggregates[-1]
    weights = []
    for level in aggregates:
        weights.append(level / total)

    return torch.cat(weights)


data = sales.merge(cal)
data = data.merge(prices)
data = data.sort_values(by=['store_id', 'item_id', 'd'])
data.index = range(data.shape[0])
data['revenue'] = data['d_'] * data['sell_price']
data['total'] = 'TOTAL'

indices = level_indices(data)
aggregates = aggregate(data, indices)
weights = level_weights(aggregates)
