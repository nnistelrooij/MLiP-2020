from pathlib import Path
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class ForecastDataset(Dataset):
    """Dataset to load forecasts.

    Attributes:
        day       = [np.ndarray] data constant per store-item
        snap      = [np.ndarray] whether or not SNAP purchases are allowed
        prices    = [np.ndarray] sell prices of each item at all stores
        sales     = [np.ndarray] unit sales of each item at all stores
        seq_len   = [int] sequence length of model input
        horizon   = [int] sequence length of model output, 0 for inference
        start_idx = [int] random start index to get different data each epoch
    """
    start_idx = 0

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

        # get data constant per store-item in one array
        self.day = np.concatenate((
            self._weekdays(calendar),
            self._weeks(calendar),
            self._monthdays(calendar),
            self._months(calendar),
            self._years(calendar),
            self._event_types(calendar)),
            axis=1
        ).astype(np.float32)

        # sort sales table for consistency
        sales = sales.sort_values(by=['store_id', 'item_id'])
        sales.index = range(sales.shape[0])

        # get data different per store-item
        self.snap = self._snap(calendar, sales)
        self.prices = self._sell_prices(calendar, prices)
        self.sales = self._unit_sales(sales)

        # normalize inputs to unit Gaussian distribution
        self.day = (self.day - 1.4436644) / 6.093091
        self.snap = (self.snap - 2.029751) / 3.42601
        self.prices = (self.prices - 2.029751) / 3.42601
        self.sales = (self.sales - 2.029751) / 3.42601

        self.seq_len = seq_len
        self.horizon = horizon

    @staticmethod
    def _weekdays(calendar):
        """One-hot representation of weekdays of shape (days, 7)."""
        return pd.RangeIndex(1, 8) == calendar[['wday']]

    @staticmethod
    def _weeks(calendar):
        """Integers of week numbers of shape (days, 1)."""
        return calendar[['wm_yr_wk']].apply(lambda x: x % 100)

    @staticmethod
    def _monthdays(calendar):
        """Integers of month days of shape (days, 1)."""
        return calendar[['date']].applymap(lambda x: x[-2:])

    @staticmethod
    def _months(calendar):
        """One-hot representation of months of shape (days, 12)."""
        return pd.RangeIndex(1, 13) == calendar[['month']]

    @staticmethod
    def _years(calendar):
        """One-hot representation of years of shape (days, 6)."""
        return pd.RangeIndex(2011, 2017) == calendar[['year']]

    @staticmethod
    def _event_types(calendar):
        """One-hot representation of event types of shape (days, 5)."""
        # make one-hot vectors for each column in calendar table
        event_types = calendar['event_type_1'].unique()
        events1 = event_types == calendar[['event_type_1']].to_numpy()
        events2 = event_types == calendar[['event_type_2']].to_numpy()

        # sum one-hot vectors and add no-event column
        events = events1 | events2
        events[:, 0] = ~np.any(events, axis=1)

        return events

    @staticmethod
    def _snap(calendar, sales):
        """Whether SNAP purchases are allowed of shape (days, N)."""
        # determine number of groups per state
        repetitions = sales.groupby('state_id').size()

        # compute SNAP data for each state
        snap_CA = pd.concat([calendar['snap_CA']]*repetitions['CA'], axis=1)
        snap_TX = pd.concat([calendar['snap_TX']]*repetitions['TX'], axis=1)
        snap_WI = pd.concat([calendar['snap_WI']]*repetitions['WI'], axis=1)

        # concatenate SNAP data for all states
        snap = pd.concat((snap_CA, snap_TX, snap_WI), axis=1)
        return snap.to_numpy(dtype=np.float32)

    @staticmethod
    def _sell_prices(calendar, prices):
        """Sell prices for each store-item of shape (days, N)."""
        # sort prices on store-items
        prices = prices.sort_values(by=['store_id', 'item_id'])

        # pivot prices table to wide format
        prices = prices.set_index(['store_id', 'item_id'])
        prices = prices.pivot(columns='wm_yr_wk')
        prices.columns = prices.columns.droplevel()

        # fill missing data with zeros and repeat prices for all days
        prices = prices.fillna(0)
        prices = prices[calendar['wm_yr_wk']]

        # return prices in (num_days, num_groups) shape
        return prices.T.to_numpy(dtype=np.float32)

    @staticmethod
    def _unit_sales(sales):
        """Unit sales for each store-item of shape (days, N)."""
        return sales.filter(like='d_').T.to_numpy(dtype=np.float32)

    def __len__(self):
        """Returns number of items in the dataset."""
        if self.horizon:
            return (len(self.sales) - 1) // self.seq_len
        else:
            return len(self.prices) - 1

    def _get_train_item(self, idx):
        # pick random start index to get different data each epoch
        if idx == 0:
            ForecastDataset.start_idx = random.randrange(self.seq_len)

        # determine index at start and end of sequence
        idx = idx * self.seq_len + self.start_idx
        end_idx = min(idx + self.seq_len, len(self.sales) - 1)

        # get data constant per store-item
        day = self.day[idx + 1:end_idx + 1]

        # stack all data different per store-item
        items = np.stack((
            self.snap[idx + 1:end_idx + 1],
            self.prices[idx + 1:end_idx + 1],
            self.sales[idx:end_idx]),
            axis=2
        )

        # get targets in shape (N, |targets|)
        targets = self.sales[end_idx:end_idx + self.horizon].T

        return day, items, targets

    def _get_inference_item(self, idx):
        # get data constant per store-item
        day = self.day[np.newaxis, idx + 1]

        # stack only SNAP and prices data; sales has variable length
        items = np.hstack((
            self.snap[idx + 1:idx + 2].T,
            self.prices[idx + 1:idx + 2].T,
            self.sales[idx:idx + 1].T)
        )[np.newaxis]

        return day, items

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
            items   = data different per store-item of shape (seq_len, 3, N)
                snap      = booleans of shape (seq_len, N)
                prices    = floats of shape (seq_len, N)
                sales     = integers of shape (seq_len, N)
            targets = unit sales of next days of shape (N, |targets|),
                where 1 <= |targets| <= horizon

        If horizon = 0, i.e. inference mode, then sales may be empty, which
        means that items may not contian sales. So then day and items
        possibly without sales are returned

        Returns [[np.ndarray]*3]:
            day   = data constant per store-item of shape (1, 32)
                weekdays  = one-hot vector of shape (1, 7)
                weeks     = integer in range [1, 53] of shape (1, 1)
                monthdays = integer in range [1, 31] of shape (1, 1)
                months    = one-hot vector of shape (1, 12)
                years     = one-hot vector of shape (1, 6)
                events    = one-hot vector of shape (1, 5)
            items = data different per store-item of shape (1, 2, N) / (1, 3, N)
                snap      = booleans of shape (1, N)
                prices    = floats of shape (1, N)
                sales     = unit sales of previous day of shape (|sales|, N),
                    where 0 <= |sales| <= 1
        """
        if self.horizon:  # training or validation mode
            return self._get_train_item(idx)
        else:  # inference mode
            return self._get_inference_item(idx)


def data_frames(path):
    """Load the data from storage into pd.DataFrame objects.

    Args:
        path     = [str] path to folder with competition data

    Returns [[pd.DataFrame]*3]:
        calendar = [pd.DataFrame] table with data on each date
        prices   = [pd.DataFrame] sell prices per store-item for each week
        sales    = [pd.DataFrame] unit sales per store-item for each day
    """
    path = Path(path)

    calendar = pd.read_csv(path / 'calendar.csv')
    prices = pd.read_csv(path / 'sell_prices.csv')
    sales = pd.read_csv(path / 'sales_train_validation.csv')

    return calendar, prices, sales


def data_loaders(calendar, prices, sales,
                 num_days, num_val_days,
                 seq_len, horizon):
    """Load the training and validation DataLoader objects.

    Args:
        calendar     = [pd.DataFrame] table with data on each date
        prices       = [pd.DataFrame] sell prices per store-item for each week
        sales        = [pd.DataFrame] unit sales per store-item for each day
        num_days     = [int] number of days for training and validation
        num_val_days = [int] number of days to use for validation data
        seq_len      = [int] sequence length of model input
        horizon      = [int] sequence length of model output, 0 for inference

    Returns [[DataLoader]*2]:
        train_loader = train DataLoader object
        val_loader   = validation DataLoader object
    """
    # get last num_days days of data plus the extra days
    num_extra_days = calendar.shape[0] - (sales.shape[1] - 6)
    calendar = calendar.iloc[-num_days - num_extra_days:]

    # get last num_days days of sales data
    sales = pd.concat((sales.iloc[:, :6], sales.iloc[:, -num_days:]), axis=1)

    # make DataLoader with only training days
    train_sales = sales.iloc[:, :-num_val_days]
    train_loader = DataLoader(ForecastDataset(
         calendar, prices, train_sales, seq_len, horizon
    ))

    # make DataLoader with all num_days days
    val_loader = DataLoader(ForecastDataset(
        calendar, prices, sales
    ))

    return train_loader, val_loader


if __name__ == '__main__':
    from datetime import datetime

    path = ('D:\\Users\\Niels-laptop\\Documents\\2019-2020\\Machine Learning '
            'in Practice\\Competition 2\\project\\')
    calendar, prices, sales = load_data(path, -365)

    # train_loader, val_loader = data_loaders(calendar, prices, sales, 28, 8, 5)

    # calendar = pd.read_csv(path + 'calendar.csv')
    # prices = pd.read_csv(path + 'sell_prices.csv')
    # sales = pd.read_csv(path + 'sales_train_validation.csv').iloc[:, :-2]

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
    dataset = ForecastDataset(calendar, prices, sales, seq_len=1, horizon=0)
    print('Time to initialize dataset: ', datetime.now() - time)

    loader = DataLoader(dataset)
    time = datetime.now()
    for day, items, sales in loader:
        print('inference: ', day.shape[1], items.shape[1], sales.shape[1])
        pass
    print('Time to retrieve data: ', datetime.now() - time)
