from datetime import datetime

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ForecastDataset(Dataset):
    """Dataset to load forecasts."""

    def __init__(self, calendar, prices, sales):
        super(Dataset, self).__init__()

        sales = sales.sort_values(by=['store_id', 'item_id'])
        sales.index = range(sales.shape[0])
        self.prices = self._sell_prices(calendar, prices, sales)
        print('prices', self.prices.shape, self.prices.dtype)
        self.sales = self._unit_sales(sales)
        print('sales', self.sales.shape, self.sales.dtype)

    @staticmethod
    def _sell_prices(calendar, prices, sales):
        num_days = calendar.shape[0]
        num_groups = sales.shape[0]

        calendar = calendar[['wm_yr_wk', 'd']]
        calendar = calendar.iloc[np.repeat(np.arange(num_days), num_groups)]
        calendar.index = range(num_days*num_groups)

        sales = sales[['store_id', 'item_id']]
        sales = pd.concat([sales]*num_days)
        sales.index = range(num_days*num_groups)

        data = pd.concat((calendar, sales), axis=1)
        data = data.merge(prices, how='left')
        data = data.drop(columns='wm_yr_wk')
        data = data.fillna(0)
        data = data.set_index(['store_id', 'item_id'])
        data = data.pivot(columns='d')
        data.columns = data.columns.droplevel()
        data.columns = data.columns.map(lambda x: int(x[2:]))
        data = data.reindex(sorted(data.columns), axis=1)
        data = data.reset_index()
        data = data.sort_values(by=['store_id', 'item_id'])
        data.index = range(num_groups)

        return data.iloc[:, 2:].T.to_numpy()

    @staticmethod
    def _unit_sales(sales):
        return sales.filter(like='d_').T.to_numpy()


if __name__ == '__main__':
    path = ('D:\\Users\\Niels-laptop\\Documents\\2019-2020\\Machine Learning '
            'in Practice\\Competition 2\\project\\')
    calendar = pd.read_csv(path + 'calendar.csv')
    prices = pd.read_csv(path + 'sell_prices.csv')
    sales = pd.read_csv(path + 'sales_train_validation.csv')

    time = datetime.now()
    dataset = ForecastDataset(calendar, prices, sales)
    print('Time to initialize dataset: ', datetime.now() - time)
