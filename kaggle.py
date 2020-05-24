import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import data_frames, ForecastDataset
from nn import Model


def infer(model, loader):
    """Infer unit sales of next days with a trained model.

    Args:
        model       = [nn.Module] trained model
        loader      = [DataLoader] DataLoader with last year of available data

    Returns [[torch.Tensor]*2]:
        validation = sales projections of next 28 days
        evaluation = sales projections of 28 days after validation days
    """
    # initialize projections as tensor
    projections = torch.empty(len(loader), 30490)

    with torch.no_grad():
        for i, (day, items, sales) in enumerate(tqdm(loader)):
            # find missing sales in projections
            start_idx = sales.shape[1] - 2 + i
            projection = projections[start_idx:i]
            projection = projection.view(1, projection.shape[0], 30490, 1)

            # concatenate inputs
            sales = torch.cat((sales, projection), dim=1)
            items = torch.cat((items, sales), dim=-1)

            # add new projections based on old projections
            y = model(day[:, :1], items[:, :1], day[:, 1:], items[:, 1:])
            projections[i] = y

    # select validation and evaluation projections from all projections
    validation = projections[-56:-28].T
    evaluation = projections[-28:].T

    return validation, evaluation


if __name__ == '__main__':
    device = torch.device('cpu')
    num_models = 300  # number of submodels
    num_days = 365  # number of days prior the days with missing sales

    # initialize trained model on correct device
    model = Model(num_models, device)
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

    # make dataloader from data
    dataset = ForecastDataset(calendar, prices, sales, seq_len=1, horizon=0)
    loader = DataLoader(dataset)

    # run model to get sales projections
    validation, evaluation = infer(model, loader)

    # add validation projections to DataFrame
    columns = [f'F{i}' for i in range(1, 29)]
    validation = pd.DataFrame(validation.tolist(), columns=columns)
    validation = pd.concat((sales[['id']], validation), axis=1)

    # add evaluation projections to DataFrame
    eval_id_col = sales[['id']].applymap(lambda x: x[:-10] + 'evaluation')
    evaluation = pd.DataFrame(evaluation.tolist(), columns=columns)
    evaluation = pd.concat((eval_id_col, evaluation), axis=1)

    # concatenate all projections and save to storage
    projections = pd.concat((validation, evaluation), axis=0)
    projections.to_csv('submission.csv', index=False)
