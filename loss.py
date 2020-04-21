import pandas as pd
import torch

from weights import loss_weights
from scales import loss_scales
from test import level_indices, aggregate

sales = pd.read_csv(
    r'D:\Users\Niels-laptop\Documents\2019-2020\Machine Learning '
    r'in Practice\Competition 2\project\sales_train_validation.csv'
)
sales = sales.iloc[:, 1:6]
sales = sales.sort_values(by=['store_id', 'item_id'])
sales.index = range(sales.shape[0])
sales['total'] = 'TOTAL'

permutations, group_indices = level_indices(sales)

weights = loss_weights(
        r'D:\Users\Niels-laptop\Documents\2019-2020\Machine Learning in '
        r'Practice\Competition 2\project'
    )
scales = loss_scales(
        r'D:\Users\Niels-laptop\Documents\2019-2020\Machine Learning in '
        r'Practice\Competition 2\project'
    )


horizon = 5

actual_sales = torch.rand(30490, horizon)
actual_sales = aggregate(actual_sales, permutations, group_indices)

projected_sales = torch.rand(30490, horizon)
projected_sales = aggregate(projected_sales, permutations, group_indices)

squared_errors = (actual_sales - projected_sales)**2
MSE = torch.sum(squared_errors, dim=1) / horizon
RMSSE = torch.sqrt(MSE / scales)
WRMSSE = torch.sum(weights * RMSSE)
