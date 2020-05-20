import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import data_frames, ForecastDataset
from nn import Model


def infer(model, loader, window_size):
    """Infer the unit sales with the model.

    Args:
        model       = [nn.Module] trained model
        loader      = [DataLoader] dataloader with last year of available data
        window_size = [int] number of projections to take the average over

    Returns [[torch.Tensor]*2]:
        validation = sales projections of next 28 days
        evaluation = sales projects of next 28 days after validation
    """
    # initialize
    projections = torch.zeros(30490, len(loader) + window_size - 1)

    with torch.no_grad():
        for i, (day, items) in enumerate(tqdm(loader)):
            if items.shape[-1] == 2:  # use sales projections at end of data
                projection = projections[:, i - 1] / window_size
                projection = projection.view(1, 1, -1, 1).to('cpu')
                items = torch.cat((items, projection), dim=-1)

            y = model(day, items)
            projections[:, i:i + window_size] += y[0, :, :window_size]

    # select validation and evaluation projections from all projections
    eval_idx = -28 - window_size + 1
    validation = projections[:, eval_idx - 28:eval_idx]
    if window_size == 1:
        evaluation = projections[:, eval_idx:]
    else:
        evaluation = projections[:, eval_idx:-window_size + 1]

    return validation, evaluation


if __name__ == '__main__':
    device = torch.device('cpu')
    horizon = 5  # forecasting horizon
    num_models = 4000  # number of submodels
    num_days = 36  # number of days prior the days with missing sales
    window_size = 1  # how much rolling average to apply, window_size <= horizon

    # initialize trained model on correct device
    model = Model(horizon, num_models, device)
    model.load_state_dict(torch.load('models/model.pt', map_location=device))
    model.reset_hidden()
    model.eval()

    path = ('D:/Users/Niels-laptop/Documents/2019-2020/Machine '
            'Learning in Practice/Competition 2/project')
    calendar, prices, sales = data_frames(path)

    # get last 365 days of data plus the extra days
    num_extra_days = calendar.shape[0] - (sales.shape[1] - 6)
    calendar = calendar.iloc[-num_days - num_extra_days:]

    # get last 365 days of sales data
    sales = pd.concat((sales.iloc[:, :6], sales.iloc[:, -num_days:]), axis=1)
    sales = sales.sort_values(by=['store_id', 'item_id'])
    sales.index = range(sales.shape[0])

    dataset = ForecastDataset(calendar, prices, sales, seq_len=1, horizon=0)
    loader = DataLoader(dataset)

    validation, evaluation = infer(model, loader, window_size)
    columns = [f'F{i}' for i in range(1, 29)]

    validation = pd.DataFrame(validation.tolist(), columns=columns)
    validation = pd.concat((sales[['id']], validation), axis=1)

    eval_id_col = sales[['id']].applymap(lambda x: x[:-10] + 'evaluation')
    evaluation = pd.DataFrame(evaluation.tolist(), columns=columns)
    evaluation = pd.concat((eval_id_col, evaluation), axis=1)

    projections = pd.concat((validation, evaluation), axis=0)
    projections.to_csv('submission.csv', index=False)
