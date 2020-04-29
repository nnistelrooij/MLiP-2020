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
        seq_len   = [int] sequence length of model input
        horizon   = [int] sequence length of model output
    """

    def __init__(self, calendar, prices, sales, seq_len=1, horizon=1):
        """Initializes forecast dataset.

        Args:
            calendar = [pd.DataFrame] table with data on each date
            prices   = [pd.DataFrame] sell prices per store-item for each week
            sales    = [pd.DataFrame] unit sales per store-item for each day
            seq_len  = [int] sequence length of model input
            horizon  = [int] sequence length of model output, 0 for inference
        """
        super(Dataset, self).__init__()

        # sort sales table for consistency
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

        self.seq_len = seq_len
        self.horizon = horizon

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
        events1 = calendar[['event_type_1']].fillna('N/A').to_numpy()
        event_types = np.unique(events1)

        events1 = event_types == events1
        events2 = event_types == calendar[['event_type_2']].to_numpy()

        return events1 + events2

    @staticmethod
    def _snap(calendar, sales):
        """Whether SNAP purchases are allowed of shape (days, N)."""
        # determine number of groups per state
        groups = sales.groupby('state_id').groups.values()
        repetitions = [len(group) for group in groups]

        # compute SNAP data for each state
        snap_CA = pd.concat([calendar['snap_CA']]*repetitions[0], axis=1)
        snap_TX = pd.concat([calendar['snap_TX']]*repetitions[1], axis=1)
        snap_WI = pd.concat([calendar['snap_WI']]*repetitions[2], axis=1)

        # concatenate SNAP data for all states
        snap = pd.concat((snap_CA, snap_TX, snap_WI), axis=1)
        return snap.to_numpy()

    @staticmethod
    def _sell_prices(calendar, prices, sales):
        """Sell prices for each store-item of shape (days, N)."""
        # determine index sizes of tables
        num_days = calendar.shape[0]
        num_groups = sales.shape[0]

        # repeat calendar table to get (num_days*num_groups, 2) shape
        calendar = calendar[['wm_yr_wk', 'd']]
        calendar = calendar.iloc[np.repeat(np.arange(num_days), num_groups)]
        calendar.index = range(num_days*num_groups)

        # concatenate sales table to get (num_days*num_groups, 2) shape
        sales = sales[['store_id', 'item_id']]
        sales = pd.concat([sales]*num_days)
        sales.index = range(num_days*num_groups)

        # join with prices table to get prices for each day and each group
        data = pd.concat((calendar, sales), axis=1)
        data = data.merge(prices, how='left')
        data = data.drop(columns='wm_yr_wk')
        data = data.fillna(0)

        # pivot table back to wide format to get (num_groups, num_days) shape
        data = data.set_index(['store_id', 'item_id'])
        data = data.pivot(columns='d')
        data.columns = data.columns.droplevel()

        # sort columns from first to last day
        data.columns = data.columns.map(lambda x: int(x[2:]))
        data = data.reindex(sorted(data.columns), axis=1)

        # sort rows by store and item ids
        data = data.reset_index()
        data = data.sort_values(by=['store_id', 'item_id'])
        data.index = range(num_groups)

        # return prices in (num_days, num_groups) shape
        return data.iloc[:, 2:].T.to_numpy()

    @staticmethod
    def _unit_sales(sales):
        """Unit sales for each store-item of shape (days, N)."""
        return sales.filter(like='d_').T.to_numpy()

    def __len__(self):
        if self.horizon:
            return len(self.sales) - self.seq_len
        else:
            return len(self.prices) - self.seq_len

    def __getitem__(self, idx):
        """Get data for self.seq_len days and targets for self.horizon days.

        If horizon > 0, i.e. training or validation mode, the targets need to be
        returned. Sales does not have a variable length, so it can be returned
        within items. So then day, items with sales, and targets are returned.

        Returns [[np.ndarray]*3]:
            day     = data constant per store-item of shape (seq_len, 32)
                weekdays  = one-hot vectors of shape (seq_len, 7)
                weeks     = integers in range [1, 53] of shape (seq_len, 1)
                monthdays = integers in range [1, 31] of shape (seq_len, 1)
                months    = one-hot vectors of shape (seq_len, 12)
                years     = one-hot vectors of shape (seq_len, 6)
                events    = one-hot vectors of shape (seq_len, 5)
            items   = data different per store-item of shape (seq_len, N, 3)
                snap      = booleans of shape (seq_len, N)
                prices    = floats of shape (seq_len, N)
                sales     = integers of shape (seq_len, N)
            targets = unit sales of next days of shape (N, |targets|), where
                0 <= |self.sales| - seq_len - idx = |targets| <= horizon

        If horizon = 0, i.e. inference mode, then sales has a variable
        length, which  means that it needs to be returned separately. So
        then day, items without sales, and sales separately are returned.

        Returns [[np.ndarray]*3]:
            day   = data constant per store-item of shape (seq_len, 32)
                weekdays  = one-hot vectors of shape (seq_len, 7)
                weeks     = integers in range [1, 53] of shape (seq_len, 1)
                monthdays = integers in range [1, 31] of shape (seq_len, 1)
                months    = one-hot vectors of shape (seq_len, 12)
                years     = one-hot vectors of shape (seq_len, 6)
                events    = one-hot vectors of shape (seq_len, 5)
            items = data different per store-item of shape (seq_len, N, 2)
                snap      = booleans of shape (seq_len, N)
                prices    = floats of shape (seq_len, N)
            sales = unit sales of previous days of shape (|sales|, N), where
                0 <= |self.sales| - idx = |sales| <= seq_len
        """
        # determine index at end of sequence
        end_idx = idx + self.seq_len

        # concatenate data constant per store-item
        day = np.concatenate((
            self.weekdays[idx + 1:end_idx + 1],
            self.weeks[idx + 1:end_idx + 1],
            self.monthdays[idx + 1:end_idx + 1],
            self.months[idx + 1:end_idx + 1],
            self.years[idx + 1:end_idx + 1],
            self.events[idx + 1:end_idx + 1]),
            axis=1
        ).astype(np.float32)

        if self.horizon:  # training or validation mode
            # stack all data different per store-item
            items = np.stack((
                self.snap[idx + 1:end_idx + 1],
                self.prices[idx + 1:end_idx + 1],
                self.sales[idx:end_idx]),
                axis=2
            ).astype(np.float32)

            # get targets in shape (N, |targets|)
            targets = self.sales[end_idx:end_idx + self.horizon].T

            return day, items, targets.astype(np.float32)
        else:  # inference mode
            # stack only SNAP and prices data; sales has variable length
            items = np.stack((
                self.snap[idx + 1:end_idx + 1],
                self.prices[idx + 1:end_idx + 1]),
                axis=2
            ).astype(np.float32)

            # return sales separately, because it has a variable length
            sales = self.sales[idx:end_idx]

            return day, items, sales.astype(np.float32)


if __name__ == '__main__':
    from datetime import datetime

    path = ('D:\\Users\\Niels-laptop\\Documents\\2019-2020\\Machine Learning '
            'in Practice\\Competition 2\\project\\')
    calendar = pd.read_csv(path + 'calendar.csv')
    prices = pd.read_csv(path + 'sell_prices.csv')
    sales = pd.read_csv(path + 'sales_train_validation.csv')
    rng = np.random.default_rng()
    rand_idx = rng.choice(np.arange(sales.shape[0]), size=3049, replace=False)
    sales = sales.iloc[rand_idx]

    time = datetime.now()
    dataset = ForecastDataset(calendar, prices, sales, seq_len=8, horizon=5)
    print('Time to initialize dataset: ', datetime.now() - time)

    loader = DataLoader(dataset)
    time = datetime.now()
    for day, items, targets in loader:
        print('training: ', day.shape[1], items.shape[1], targets.shape[2])
        pass
    print('Time to retrieve data: ', datetime.now() - time)

    time = datetime.now()
    dataset = ForecastDataset(calendar, prices, sales, seq_len=8, horizon=0)
    print('Time to initialize dataset: ', datetime.now() - time)

    loader = DataLoader(dataset)
    time = datetime.now()
    for day, items, sales in loader:
        print('inference: ', day.shape[1], items.shape[1], sales.shape[1])
        pass
    print('Time to retrieve data: ', datetime.now() - time)
