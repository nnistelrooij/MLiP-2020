import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import ForecastDataset
from nn import Model


def infer(model, loader):
    """Infer the unit sales with the model.

    Args:
        model  = [nn.Module] trained model
        loader = [DataLoader] dataloader with last year of available data

    Returns [[torch.Tensor]*2]:
        validation = sales projections of next 28 days
        evaluation = sales projects of next 28 days after validation
    """
    projections = []
    for day, items in tqdm(loader):
        if items.shape[2] == 2:  # use sales projections at end of data
            projection = projections[-1].view(1, 1, 1, items.shape[-1])
            items = torch.cat((items, projection.to('cpu')), dim=2)

        y = model(day, items)
        projections.append(y[:, 0])
        
    validation = torch.stack(projections[-56:-28], dim=1)
    evaluation = torch.stack(projections[-28:], dim=1)

    return validation, evaluation


if __name__ == '__main__':
    device = torch.device('cuda')
    horizon = 5  # forecasting horizon
    num_models = 1000

    model = Model(horizon, num_models, device, True)
    # model.load_state_dict(torch.load('models/model.pt', map_location=device))
    model.eval()

    path = ('D:\\Users\\Niels-laptop\\Documents\\2019-2020\\Machine Learning '
            'in Practice\\Competition 2\\project\\')
    calendar = pd.read_csv(path + r'\calendar.csv').iloc[-365:]
    sales = pd.read_csv(path + r'\sales_train_validation.csv')
    sales = pd.concat((sales.iloc[:, :6], sales.iloc[:, -365+28+28:]), axis=1)
    sales = sales.sort_values(by=['store_id', 'item_id'])
    prices = pd.read_csv(path + r'\sell_prices.csv')

    dataset = ForecastDataset(calendar, prices, sales, seq_len=1, horizon=0)
    loader = DataLoader(dataset)

    validation, evaluation = infer(model, loader)
    columns = [f'F{i}' for i in range(1, 29)]

    validation = pd.DataFrame(validation.tolist(), columns=columns)
    validation = pd.concat((sales[['id']], validation), axis=1)

    evaluation = pd.DataFrame(evaluation.tolist(), columns=columns)
    eval_col = sales[['id']].applymap(lambda x: x[:-10] + 'evaluation')
    evaluation = pd.concat((eval_col, evaluation), axis=1)

    projections = pd.concat((validation, evaluation), axis=0)
    projections.to_csv('submission.csv', index=False)
