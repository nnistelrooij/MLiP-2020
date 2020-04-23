from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam

from nn import WRMSSE


class MLPs(nn.Module):

    def __init__(self, device, groups, horizon=5):
        super(MLPs, self).__init__()

        # self.mlps = nn.ModuleList()
        # for _ in range(groups):
        #     self.mlps.append(nn.Linear(11, horizon))

        self._device = device
        self.to(self._device)

        self.horizon = horizon

    def forward(self, x):
        x = x.to(self._device)

        # y = []
        # for group, mlp in zip(x, self.mlps):
        #     y.append(mlp(group))
        # y = torch.stack(y)

        x = x.flatten()
        y = self.mlp(x)
        y = y.view(-1, self.horizon)

        return y


if __name__ == '__main__':
    import pandas as pd

    device = torch.device('cuda')
    groups = 50
    model = MLPs(device, groups)

    optimizer = Adam(model.parameters(), lr=0.01)

    path = ('D:\\Users\\Niels-laptop\\Documents\\2019-2020\\Machine Learning '
            'in Practice\\Competition 2\\project\\')
    calendar = pd.read_csv(path + 'calendar.csv')
    prices = pd.read_csv(path + 'sell_prices.csv')
    sales = pd.read_csv(path + 'sales_train_validation.csv').iloc[:groups]
    criterion = WRMSSE(device, calendar, prices, sales)

    print('[')
    for _ in range(100):
        x = torch.rand(groups, 11)
        t = torch.full((groups, 5), 4.0)

        time = datetime.now()
        y = model(x)
        # print(y[0])
        loss = criterion(y, t)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print((datetime.now() - time).microseconds, ', ')
    print(']')


