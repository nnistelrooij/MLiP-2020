import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class ForecastDataset(Dataset):
    """Dataset to load forecasts.

    Attributes:
        weekdays  = [np.ndarray] one-hot vectors of weekdays
        weeks     = [np.ndarray] integers of weeks in range [1, 53]
        monthdays = [np.ndarray] integers of monthdays in range [1, 31]
        months    = [np.ndarray] one-hot vectors of months
        years     = [np.ndarray] one-hot vectors of years
        events    = [np.ndarray] one-hot vectors of event types
        snap      = [np.ndarray] whether or not SNAP purchases are allowed
        prices    = [np.ndarray] sell prices of each item at all stores
        sales     = [np.ndarray] unit sales of each item at all store
    """

    def __init__(self, calendar, prices, sales):
        """Initializes forecast dataset.

        Args:
            calendar = [pd.DataFrame] table with data on each date
            prices   = [pd.DataFrame] table with average sell prices each week
            sales    = [pd.DataFrame] table with sales per item for each day
        """
        super(Dataset, self).__init__()

        sales = sales.sort_values(by=['store_id', 'item_id'])
        sales.index = range(sales.shape[0])

        self.weekdays = self._weekdays(calendar)
        self.weeks = self._weeks(calendar)
        self.monthdays = self._monthdays(calendar)
        self.months = self._months(calendar)
        self.years = self._years(calendar)
        self.events = self._event_types(calendar)
        self.snap = self._snap(calendar, sales)
        self.prices = self._sell_prices(calendar, prices, sales)
        self.sales = self._unit_sales(sales)

    @staticmethod
    def _weekdays(calendar):
        """One-hot representation of weekdays of shape (days, 7)."""
        return np.arange(1, 8) == calendar[['wday']].to_numpy()

    @staticmethod
    def _weeks(calendar):
        """Integers of week numbers of shape (days, 1)."""
        return calendar[['wm_yr_wk']].apply(lambda x: x % 100).to_numpy()

    @staticmethod
    def _monthdays(calendar):
        """Integers of month days of shape (days, 1)."""
        return calendar[['date']].applymap(lambda x: int(x[-2:])).to_numpy()

    @staticmethod
    def _months(calendar):
        """One-hot representation of months of shape (days, 12)."""
        return np.arange(1, 13) == calendar[['month']].to_numpy()

    @staticmethod
    def _years(calendar):
        """One-hot representation of years of shape (days, 6)."""
        return np.arange(2011, 2017) == calendar[['year']].to_numpy()

    @staticmethod
    def _event_types(calendar):
        """One-hot representation of event types of shape (days, 5)."""
        events1 = calendar[['event_type_1']].fillna('N/A')
        event_types = events1['event_type_1'].unique()

        events1 = event_types == events1.to_numpy()
        events2 = event_types == calendar[['event_type_2']].to_numpy()

        return events1 + events2

    @staticmethod
    def _snap(calendar, sales):
        """Whether SNAP purchases are allowed of shape (days, N)."""
        groups = sales.groupby('state_id').groups.values()
        repetitions = [len(group) for group in groups]

        snap_CA = pd.concat([calendar['snap_CA']]*repetitions[0], axis=1)
        snap_TX = pd.concat([calendar['snap_TX']]*repetitions[1], axis=1)
        snap_WI = pd.concat([calendar['snap_WI']]*repetitions[2], axis=1)

        snap = pd.concat((snap_CA, snap_TX, snap_WI), axis=1)
        return snap.to_numpy()

    @staticmethod
    def _sell_prices(calendar, prices, sales):
        """Sell prices for each store-item of shape (days, N)."""
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

        return data.loc[:, 1:].T.to_numpy()

    @staticmethod
    def _unit_sales(sales):
        """Unit sales for each store-item of shape (days, N)."""
        return sales.filter(like='d_').T.to_numpy()

    def __len__(self):
        return len(self.sales)

    def __getitem__(self, idx):
        """Gets all data for one day.

        Returns [[np.ndarray]*2]:
            day = [[float]*32] data constant for all store-item groups
                weekday  = [[float]*7] one-hot vector of weekday
                week     = [float] integer of week in range [1, 53]
                monthday = [float] integer of monthday in range [1, 31]
                month    = [[float]*12] one-hot vector of month
                year     = [[float]*6] one-hot vector of year
                event    = [[float]*5] one-hot vector of event type
            items = [[[float]*N]*3] data different for each store-item group
                snap  = [[float]*N] whether or not SNAP purchases are allowed
                price = [[float]*N] sell price of each item at all stores
                sales = [[float]*N] unit sales of each item at all stores
        """
        day = np.concatenate((
            self.weekdays[idx],
            self.weeks[idx],
            self.monthdays[idx],
            self.months[idx],
            self.years[idx],
            self.events[idx],
        )).astype(np.float32)

        items = np.stack((
            self.snap[idx],
            self.prices[idx],
            self.sales[idx]),
            axis=1
        ).astype(np.float32)

        return day, items


if __name__ == '__main__':
    from datetime import datetime

    path = ('D:\\Users\\Niels-laptop\\Documents\\2019-2020\\Machine Learning '
            'in Practice\\Competition 2\\project\\')
    calendar = pd.read_csv(path + 'calendar.csv')
    prices = pd.read_csv(path + 'sell_prices.csv')
    sales = pd.read_csv(path + 'sales_train_validation.csv')

    time = datetime.now()
    dataset = ForecastDataset(calendar, prices, sales)
    print('Time to initialize dataset: ', datetime.now() - time)

    loader = DataLoader(dataset)
    time = datetime.now()
    for day, items in loader:
        pass
    print('Time to retrieve data: ', datetime.now() - time)
