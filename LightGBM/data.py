import numpy as np
import pandas as pd


def downcast(df, verbose=False):
    """
    Downcast the data to reduce memory usage.
    Adapted from: https://www.kaggle.com/ragnar123/very-fst-model

    Args:
        df = [pd.DataFrame] pandas dataframe
    
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
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def obj_as_cat_int(df, ignore=[]):
    """
    Convert object columns to categorical integers.

    Args:
        df = [pd.DataFrame] pandas dataframe
    
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


def optimize_df(calendar, prices, sales, days=None, verbose=False):
    """
    Optimize dataframe.

    Args:
        calendar = [pd.DataFrame] dates of product sales
        prices   = [pd.DataFrame] price of the products sold per store and date
        sales    = [pd.DataFrame] historical daily unit sales data per product and store 
        days     = [int] number of days to keep
        verbose  = [boolean] if True, print memory reduction

    Returns [(pd.DataFrame) * 3]
        Optimized dataframes.s
    """
    assert days > 56, "Minimum days is 57."
    assert days < 1914, "Maximum days is 1913."
    calendar['date'] = pd.to_datetime(calendar['date'], format='%Y-%m-%d')

    if days:
        sales.drop(sales.columns[6:-days], axis=1, inplace=True)

    calendar = downcast( obj_as_cat_int(calendar, ignore=['d']), verbose )
    prices = downcast( obj_as_cat_int(prices), verbose )
    sales = downcast( obj_as_cat_int(sales, ignore=['id']), verbose )

    return calendar, prices, sales


def features(df):
    """
    Create features.
    Adapted from: https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50

    Args:
        df = [pd.DataFrame] wide format dataframe
    """
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df[["id", "sales"]].groupby("id")["sales"].shift(lag)

    windows = [7, 28]
    for window in windows :
        for lag,lag_col in zip(lags, lag_cols):
            df[f"rmean_{lag}_{window}"] = df[["id", lag_col]].groupby("id")[lag_col].transform(lambda x: x.rolling(window).mean())

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


if __name__ == "__main__":
    pass
