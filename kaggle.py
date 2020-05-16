import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import ForecastDataset
from model import Model


def tes_t(model, loader):
    projections = []
    for day, items in tqdm(loader):
        if items.shape[2] == 2:
            projection = projections[-1].view(1, 1, 1, items.shape[-1])
            items = torch.cat((items, projection.to('cpu')), dim=2)

        y = model(day, items)
        projections.append(y[:, 0])

    return torch.stack(projections[-56:-28], dim=1), torch.stack(projections[-28:], dim=1)


if __name__ == '__main__':
    device = torch.device('cuda')
    num_const = 32  # number of inputs per sub-LSTM that are constant per store-item
    num_var = 3  # number of inputs per sub-LSTM that are different per store-item
    horizon = 5  # number of hidden units per sub-LSTM and output of the entire model (= forecasting horizon)
    num_groups = 30490  # number of store-item groups

    model = Model(num_const, num_var, horizon, horizon, num_groups, 1000, device)
    model.to(device)
    model.eval()
    # model.load_state_dict(torch.load('model.pt', map_location=device))

    # path = ('D:\\Users\\Niels-laptop\\Documents\\2019-2020\\Machine Learning '
    #         'in Practice\\Competition 2\\project\\')
    path = r'C:\Users\Niels\Downloads\MLiP-2020'
    calendar = pd.read_csv(path + r'\calendar.csv').iloc[-365:]
    sales = pd.read_csv(path + r'\sales_train_validation.csv')
    sales = pd.concat((sales.iloc[:, :6], sales.iloc[:, -365+28+28:]), axis=1)
    sales = sales.sort_values(by=['store_id', 'item_id'])
    prices = pd.read_csv(path + r'\sell_prices.csv')

    dataset = ForecastDataset(calendar, prices, sales, seq_len=1, horizon=0)
    loader = DataLoader(dataset)

    validation, evaluation = tes_t(model, loader)
    columns = [f'F{i}' for i in range(1, 29)]

    validation = pd.DataFrame(validation.tolist(), columns=columns)
    validation = pd.concat((sales[['id']], validation), axis=1)

    evaluation = pd.DataFrame(evaluation.tolist(), columns=columns)
    eval_col = sales[['id']].applymap(lambda x: x[:-10] + 'evaluation')
    evaluation = pd.concat((eval_col, evaluation), axis=1)

    projections = pd.concat((validation, evaluation), axis=0)
    projections.to_csv('submission.csv', index=False)
