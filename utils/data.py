import math
from pathlib import Path
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class ForecastDataset(Dataset):
    """Dataset to load forecasts.

    Attributes:
        start_idx  = [int] random start index to get different data each epoch
        num_const  = [int] number of inputs constant per store-item group
        num_var    = [int] number of inputs different per store-item group
        num_groups = [int] number of store-item groups
        day        = [np.ndarray] data constant per store-item group
        snap       = [np.ndarray] whether or not SNAP purchases are allowed
        prices     = [np.ndarray] sell prices of each item at all stores
        sales      = [np.ndarray] unit sales of each item at all stores
        seq_len    = [int] sequence length of model input
        horizon    = [int] sequence length of model output, 0 for inference
    """
    start_idx = 0
    num_const = 0
    num_var = 0
    num_groups = 0

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

        # get data different per store-item
        self.snap = self._snap(calendar, sales)
        self.prices = self._sell_prices(calendar, prices)
        self.sales = self._unit_sales(sales)

        ForecastDataset.num_const = self.day.shape[1]
        ForecastDataset.num_var = 3
        ForecastDataset.num_groups = self.sales.shape[1]
        self.seq_len = seq_len
        self.horizon = horizon

    @staticmethod
    def _weekdays(calendar):
        """One-hot representations of weekdays of shape (days, 7)."""
        return pd.RangeIndex(1, 8) == calendar[['wday']]

    @staticmethod
    def _weeks(calendar):
        """One-hot representations of week numbers of shape (days, 1)."""
        return calendar[['wm_yr_wk']].apply(lambda x: x % 100)

    @staticmethod
    def _monthdays(calendar):
        """One-hot representations of month days of shape (days, 1)."""
        return calendar[['date']].applymap(lambda x: x[-2:])

    @staticmethod
    def _months(calendar):
        """One-hot representations of months of shape (days, 12)."""
        return pd.RangeIndex(1, 13) == calendar[['month']]

    @staticmethod
    def _years(calendar):
        """One-hot representations of years of shape (days, 6)."""
        return pd.RangeIndex(2011, 2017) == calendar[['year']]

    @staticmethod
    def _event_types(calendar):
        """One-hot representations of event types of shape (days, 5)."""
        # make one-hot vectors for each event type column in calendar table
        event_types = calendar['event_type_1'].unique()
        events1 = event_types == calendar[['event_type_1']].to_numpy()
        events2 = event_types == calendar[['event_type_2']].to_numpy()

        # sum one-hot vectors and add no-event column
        events = events1 | events2
        events[:, 0] = ~np.any(events, axis=1)

        return events

    @staticmethod
    def _snap(calendar, sales):
        """Whether SNAP purchases are allowed of shape (days, 30490)."""
        # determine number of groups per state
        repetitions = sales.groupby('state_id').size()

        # compute SNAP data for each state
        snap_CA = pd.concat([calendar['snap_CA']]*repetitions['CA'], axis=1)
        snap_TX = pd.concat([calendar['snap_TX']]*repetitions['TX'], axis=1)
        snap_WI = pd.concat([calendar['snap_WI']]*repetitions['WI'], axis=1)

        # concatenate SNAP data for all states
        snap = pd.concat((snap_CA, snap_TX, snap_WI), axis=1)

        # return normalized SNAP data in (num_days, num_groups) shape
        snap = (snap - 0.33011681056373793) / 0.470254932932085
        return snap.to_numpy(dtype=np.float32)

    @staticmethod
    def _sell_prices(calendar, prices):
        """Sell prices for each store-item group of shape (days, 30490)."""
        # pivot prices table to wide format
        prices = prices.set_index(['store_id', 'item_id'])
        prices = prices.pivot(columns='wm_yr_wk')
        prices.columns = prices.columns.droplevel()

        # fill missing data with zeros and repeat prices for all days
        prices = prices.fillna(0)
        prices = prices[calendar['wm_yr_wk']]

        # return normalized prices in (num_days, num_groups) shape
        prices = (prices - 3.507094282300549) / 3.5222671176612437
        return prices.T.to_numpy(dtype=np.float32)

    @staticmethod
    def _unit_sales(sales):
        """Unit sales for each store-item group of shape (days, 30490)."""
        return sales.filter(like='d_').T.to_numpy(dtype=np.float32)

    def __len__(self):
        """Returns number of items in the dataset."""
        if self.horizon:
            return (len(self.sales) - 2) // self.seq_len
        else:
            return len(self.prices) - 2

    def _get_train_validation_item(self, idx):
        """Get input and target data during training or validation.

        Each epoch, a different starting index is sampled to increase data
        diversity.
        The actual size of the returned arrays might not be exactly seq_len
        and horizon + 1, but since the model can handle an arbitrary sequence
        length, that is not really a problem.

        Returns [[np.ndarray]*4]:
            day   = input data constant per store-item group
            targets_day = target data constant per store-item group
                Shapes are (seq_len, 114), respectively
                (horizon + 1, 114) with constituents:

                weekdays  = one-hot vectors of shape (T, 7)
                weeks     = one-hot vectors of shape (T, 53)
                monthdays = one-hot vectors of shape (T, 31)
                months    = one-hot vectors of shape (T, 12)
                years     = one-hot vectors of shape (T, 6)
                events    = one-hot vectors of shape (T, 5)
            items   = input data different per store-item group
            targets_items = target data different per store-item group
                Shapes are (seq_len, 30490, 3), respectively
                (horizon + 1, 30490, 3) with constituents:

                snap      = booleans of shape (T, 30490)
                prices    = floats of shape (T, 30490)
                sales     = integers of shape (T, 30490)
        """
        # pick random start index to get different data each epoch
        if idx == 0:
            ForecastDataset.start_idx = random.randrange(self.seq_len)

        # determine index at start and end of sequence
        idx = idx * self.seq_len + self.start_idx
        end_idx = min(idx + self.seq_len, len(self.sales) - 2)
        targets_end_idx = min(end_idx + self.horizon + 1, len(self.sales))

        # get data constant per store-item group
        day = self.day[idx + 1:end_idx + 1]
        targets_day = self.day[end_idx + 1:targets_end_idx + 1]

        # stack all data different per store-item group
        items = np.stack((
            self.snap[idx + 1:end_idx + 1],
            self.prices[idx + 1:end_idx + 1],
            self.sales[idx:end_idx]),
            axis=2
        )
        targets_items = np.stack((
            self.snap[end_idx + 1:targets_end_idx + 1],
            self.prices[end_idx + 1:targets_end_idx + 1],
            self.sales[end_idx:targets_end_idx]),
            axis=2
        )

        return day, targets_day, items, targets_items

    def _get_inference_item(self, idx):
        """Get input data during inference.

        If horizon = 0, i.e. inference mode, then sales may be empty. So day,
        items, and sales separately, are returned.

        Returns [[np.ndarray]*3]:
            day   = data constant per store-item group of shape (2, 114)
                weekdays  = one-hot vector of shape (2, 7)
                weeks     = one-hot vectors of shape (2, 53)
                monthdays = one-hot vectors of shape (2, 31)
                months    = one-hot vector of shape (2, 12)
                years     = one-hot vector of shape (2, 6)
                events    = one-hot vector of shape (2, 5)
            items = data different per store-item group of shape (2, 30490, 2)
                snap      = booleans of shape (2, 30490)
                prices    = floats of shape (2, 30490)
            sales = sales of previous days per store-item group
                The shape is (T, 30490, 1), where 0 <= T <= 2.
        """
        # get data constant per store-item
        day = self.day[idx + 1:idx + 3]

        # stack only SNAP and prices data; sales may have different length
        items = np.stack((
            self.snap[idx + 1:idx + 3],
            self.prices[idx + 1:idx + 3]),
            axis=2
        )

        # get sales, which may be empty
        sales = self.sales[idx:idx + 2, :, np.newaxis]

        return day, items, sales

    def __getitem__(self, idx):
        if self.horizon:  # training or validation mode
            return self._get_train_validation_item(idx)
        else:  # inference mode
            return self._get_inference_item(idx)


def data_frames(path):
    """Load the data from storage into pd.DataFrame objects.

    Args:
        path = [str] path to folder with competition data

    Returns [[pd.DataFrame]*3]:
        calendar = [pd.DataFrame] sorted table with data on each date
        prices   = [pd.DataFrame] sell prices per store-item for each week
        sales    = [pd.DataFrame] unit sales per store-item for each day
    """
    path = Path(path)

    # load DataFrames from storage
    calendar = pd.read_csv(path / 'calendar.csv')
    prices = pd.read_csv(path / 'sell_prices.csv')
    sales = pd.read_csv(path / 'sales_train_validation.csv')

    # sort data on lowest level of time series hierarchy
    prices = prices.sort_values(by=['store_id', 'item_id'])
    prices.index = range(prices.shape[0])
    sales = sales.sort_values(by=['store_id', 'item_id'])
    sales.index = range(sales.shape[0])

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
    calendar, prices, sales = data_frames(path)

    time = datetime.now()
    dataset = ForecastDataset(calendar, prices, sales, seq_len=8, horizon=5)
    print('Time to initialize dataset: ', datetime.now() - time)

    loader = DataLoader(dataset)
    time = datetime.now()
    for day, items, targets_day, targets_items in loader:
        print('training: ', day.shape[1], items.shape[1],
              targets_day.shape[1], targets_items.shape[1])
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
