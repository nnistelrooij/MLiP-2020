import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

import lightgbm as lgb


# Global constants
MAX_LAG = 57


def downcast(df, verbose=False):
    """
    Downcast the data to reduce memory usage.  
    Adapted from: https://www.kaggle.com/ragnar123/very-fst-model

    Args:
        df      = [pd.DataFrame] pandas dataframe
        verbose = [boolean] if True, print memory reduction
    
    Returns [pd.DataFrame]:
        Downcasted data.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2 

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({(start_mem - end_mem) / start_mem:.1%} reduction)')
    return df


def obj_as_cat_int(df, ignore=[]):
    """
    Convert object columns to categorical integers.

    Args:
        df      = [pd.DataFrame] pandas dataframe
        ignore  = [list] list of columns to ignore in conversion
    
    Returns [pd.DataFrame]:
        Data where object columns are encoded as categorical integers.
    """
    obj_cols = df.select_dtypes(include='object').columns
    for col in obj_cols:
        if col not in ignore:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes.astype("int16")
            df[col] -= df[col].min()
    return df


def optimize_df(calendar, prices, sales, days=None, val_days=0, verbose=False):
    """
    Optimize dataframe.

    Args:
        calendar = [pd.DataFrame] dates of product sales
        prices   = [pd.DataFrame] price of the products sold per store and date
        sales    = [pd.DataFrame] historical daily unit sales data per product and store 
        days     = [int] number of days to keep
        val_days = [int] number of validation days
        verbose  = [boolean] if True, print memory reduction

    Returns [[pd.DataFrame] * 3]
        Optimized dataframes.
    """
    assert days > 56, f"Minimum days is {MAX_LAG}."
    assert val_days != 0, "Invalid number of validation days."
    calendar['date'] = pd.to_datetime(calendar['date'], format='%Y-%m-%d')

    if val_days:
        sales = sales.drop(sales.columns[-val_days:], axis=1)
    if days:        
        sales = sales.drop(sales.columns[6:-days], axis=1)

    calendar = downcast( obj_as_cat_int(calendar, ignore=['d']), verbose )
    prices = downcast( obj_as_cat_int(prices), verbose )
    sales = downcast( obj_as_cat_int(sales, ignore=['id']), verbose )

    return calendar, prices, sales


def melt_and_merge(calendar, prices, sales, submission=False):
    """
    Convert sales from wide to long format, and merge sales with
    calendar and prices to create one dataframe.

    Args:
        calendar    = [pd.DataFrame] dates of product sales
        prices      = [pd.DataFrame] price of the products sold per store and date
        sales       = [pd.DataFrame] historical daily unit sales data per product and store
        submission  = [boolean] if True, add day columns required for submission

    Returns [pd.DataFrame]:
        Merged long format dataframe.
    """
    id_cols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

    if submission:
        last_day = int(sales.columns[-1].replace('d_', ''))
        sales.drop(sales.columns[6:-MAX_LAG], axis=1, inplace=True)
        for day in range(last_day + 1, last_day + 28 + 1):
            sales[f'd_{day}'] = np.nan

    df = pd.melt(sales,
                id_vars=id_cols,
                var_name='d',
                value_name='sales')

    df = df.merge(calendar, on='d', copy = False)
    df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], copy=False)

    return df


def features(df, submission=False):
    """
    Create lag and rolling mean features.
    Adapted from: https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50

    Args:
        df          = [pd.DataFrame] long format dataframe
        submission  = [boolean] if True, do not drop NaN rows

    Returns [pd.DataFrame]:
        Dataframe with created features.
    """
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df[["id", "sales"]].groupby("id")["sales"].shift(lag)

    windows = [7, 28]
    for window in windows :
        for lag, lag_col in zip(lags, lag_cols):
            lag_by_id = df[["id", lag_col]].groupby("id")[lag_col]
            df[f"rmean_{lag}_{window}"] = lag_by_id.transform(lambda x: x.rolling(window).mean())

    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day"
    }
    
    for name, attribute in date_features.items():
        if name in df.columns:
            df[name] = df[name].astype("int16")
        else:
            df[name] = getattr(df["date"].dt, attribute).astype("int16")

    if not submission:
        df.dropna(inplace=True)

    return df


def training_data(df):
    """
    Split data into features and labels for training.

    Args:
        df = [pd.DataFrame] pandas dataframe

    Returns [[pd.DataFrame] * 2]:
        X = training features
        y = training labels
    """
    drop_cols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]
    keep_cols = df.columns[~df.columns.isin(drop_cols)]

    X = df[keep_cols]
    y = df["sales"]

    return X, y


def lgb_dataset(calendar, prices, sales):
    """
    Make LightGBM training and validation datasets from preprocessed dataframes.  
    NOTE: preprocessed means that categorical features have been converted to integers. 

    Args:
        calendar = [pd.DataFrame] dates of product sales
        prices   = [pd.DataFrame] price of the products sold per store and date
        sales    = [pd.DataFrame] historical daily unit sales data per product and store 

    Returns [[lgb.Dataset] * 2]:
        train_set = LightGBM training dataset
        val_set = LightGBM validation dataset
    """
    df = melt_and_merge(calendar, prices, sales)
    df = features(df)
    
    X, y = training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    cat_features = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id'] + \
                   ['event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']

    train_set = lgb.Dataset(X_train, 
                            label=y_train, 
                            categorical_feature=cat_features)

    val_set = lgb.Dataset(X_test, 
                          label=y_test,
                          categorical_feature=cat_features)

    return train_set, val_set


def data_frames(path):
    """
    Load the data from storage into pd.DataFrame objects.

    Args:
        path = [str] path to folder with competition data

    Returns [[pd.DataFrame] * 3]:
        calendar = [pd.DataFrame] dates of product sales
        prices   = [pd.DataFrame] price of the products sold per store and date
        sales    = [pd.DataFrame] historical daily unit sales data per product and store 
    """
    path = Path(path)

    calendar = pd.read_csv(path / 'calendar.csv')
    prices = pd.read_csv(path / 'sell_prices.csv')
    sales = pd.read_csv(path / 'sales_train_validation.csv')

    return calendar, prices, sales


if __name__ == "__main__":
    pass
