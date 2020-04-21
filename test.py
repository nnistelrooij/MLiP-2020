import pandas as pd
import numpy as np
import torch


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

            group_sizes = [len(group) for group in permutation]
            group_end_indices = np.cumsum(group_sizes) - 1
            group_indices.append(group_end_indices)

            permutation = np.concatenate(permutation)
            permutations.append(permutation)

    return permutations, group_indices


def aggregate(sales, permutations, group_indices):
    aggregates = []
    for permutation, group_end_indices in zip(permutations, group_indices):
        permutation = sales[permutation]
        sums1 = permutation.cumsum(0)[group_end_indices]
        sums2 = torch.cat((torch.zeros_like(sales[:1]), sums1[:-1]))

        aggregates.append(sums1 - sums2)

    return torch.cat(aggregates)


if __name__ == '__main__':
    sales = pd.read_csv(
        r'D:\Users\Niels-laptop\Documents\2019-2020\Machine Learning '
        r'in Practice\Competition 2\project\sales_train_validation.csv'
    )
    sales = sales.iloc[:, 1:6]
    sales = sales.sort_values(by=['store_id', 'item_id'])
    sales.index = range(sales.shape[0])
    sales['total'] = 'TOTAL'

    permutations, group_indices = level_indices(sales)

    tensor = torch.rand(30490, 3)
    aggregates = aggregate(tensor, permutations, group_indices)
