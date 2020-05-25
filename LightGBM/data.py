import numpy as np
import pandas as pd


def downcast(df, verbose=False):
    """
    Downcasts the data to reduce memory usage.
    Adapted from: https://www.kaggle.com/kyakovlev/m5-simple-fe

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
        elif col == 'date':
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
        elif str(col_type) == 'object':
            df[col] = df[col].astype('category')
        else:
            print(f'Unexpected dtype {col_type} in dataframe.')   

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def cat_to_int(df, ignore=[]):
    """
    Convert categorical columns to integer.

    Args:
        df = [pd.DataFrame] pandas dataframe
    
    Returns [pd.DataFrame]:
        Data where categorical columns are encoded as integers.
    """
    cat_cols = df.select_dtypes(include='category').columns
    for col in cat_cols:
        if col not in ignore:
            df[col] = df[col].cat.codes.astype("int16")
            df[col] -= df[col].min()
    return df


if __name__ == "__main__":
    pass
