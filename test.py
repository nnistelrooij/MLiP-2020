import pandas as pd
import numpy as np
import torch


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


def aggregate(sales, indices):
    aggregates = []
    for permutation, group_indices in indices:
        permutation = sales[permutation]
        sums1 = permutation.cumsum(0)[group_indices]
        sums2 = torch.cat((torch.zeros_like(sales[:1]), sums1[:-1]))

        aggregates.append(sums1 - sums2)

    return aggregates


sales = pd.read_csv(
    r'D:\Users\Niels-laptop\Documents\2019-2020\Machine Learning '
    r'in Practice\Competition 2\project\sales_train_validation.csv'
)
sales['total'] = 'TOTAL'

indices = level_indices(sales)

tensor = torch.rand(30490, device=torch.device('cuda'), requires_grad=True)
aggregates = aggregate(tensor, indices)
