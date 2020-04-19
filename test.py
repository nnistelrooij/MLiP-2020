import pandas as pd
import numpy as np
import torch


def indices(df):
    permutations = []
    group_indices = []

    groups1 = ['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id', 'total']
    groups2 = [['', ' cat_id', ' dept_id', ' item_id']]*2 + [['']]*4
    for group1, group2 in zip(groups1, groups2):
        for group2 in group2:
            group_keys = (group1 + group2).split()
            groups = df.groupby(group_keys)

            permutation = list(groups.indices.values())
            permutation.sort(key=lambda x: x[0])

            group_sizes = [len(group) for group in permutation]
            group_indices.append(np.cumsum(group_sizes) - 1)

            permutation = np.concatenate(permutation)
            permutations.append(permutation)

    return permutations, group_indices


def aggregate(sales, permutations, group_indices):
    aggregate = []
    for permutation, group_idx in zip(permutations, group_indices):
        permutation = sales[permutation]
        sums1 = permutation.cumsum(0)[group_idx]
        sums2 = torch.cat((torch.zeros_like(sales[:1]), sums1[:-1]))

        aggregate.append(sums1 - sums2)

    return aggregate



sales = pd.read_csv(r'D:\Users\Niels-laptop\Documents\2019-2020\Machine Learning in Practice\Competition 2\project\sales_train_validation.csv')
sales['total'] = 'TOTAL'
permutations, group_indices = indices(sales)


tensor = torch.rand(30490, device=torch.device('cuda'), requires_grad=True)

levels = aggregate(tensor, permutations, group_indices)

i = 3
