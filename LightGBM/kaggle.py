import pandas as pd

import lightgbm as lgb

from datetime import datetime, timedelta
from tqdm import tqdm

from data import data_frames, optimize_df, melt_and_merge, features, lgb_dataset
import model


# Global constants
FIRST_DATE = datetime(2016, 4, 25) #  first date of validation and evaluation days
MAX_LAG = timedelta(days=57)


def next_day_features(df, forecast_date):
    """
    Create features of the next day to forecast.

    Args:
        df            = [pd.DataFrame] long format dataframe
        forecast_date = [datetime] forecast date

    Returns [pd.DataFrame]:
        Dataframe with features for the next day to forecast.
    """
    forecast_df = df[ (df['date'] >= forecast_date - MAX_LAG) & (df['date'] <= forecast_date) ].copy()    
    forecast_df = features(forecast_df, submission=True)   
    return forecast_df


def make_submission(df):
    """
    Create dataframe in the correct format for submission.

    Args:
        df = [pd.DataFrame] pandas dataframe

    Returns [pd.DataFrame]:
        Submission dataframe.
    """
    cols = [f"F{i}" for i in range(1, 29)] 

    submission = df.loc[df['date'] >= FIRST_DATE, ['id', 'sales']].copy()
    submission['F'] = [f'F{rank}' for rank in submission.groupby('id')['id'].cumcount() + 1]
    submission = submission.set_index(['id', 'F']).unstack()['sales'][cols].reset_index()
    submission.fillna(0., inplace=True)
    submission.sort_values("id", inplace=True)
    submission.reset_index(drop=True, inplace=True)

    # make a dummy evaluation forecast
    submission_eval = submission.copy()
    submission_eval['id'] = submission_eval['id'].str.replace('validation', 'evaluation')
    submission = pd.concat([submission, submission_eval], axis=0, sort=False)    

    return submission


def infer(model, calendar, prices, sales):
    """
    Infer the unit sales with the model.

    Args:
        model    = [lgb.Booster] trained LightGBM model
        calendar = [pd.DataFrame] dates of product sales
        prices   = [pd.DataFrame] price of the products sold per store and date
        sales    = [pd.DataFrame] historical daily unit sales data per product and store 

    Returns [pd.DataFrame]:
        Submission dataframe.
    """
    # create test dataset for submission
    df = melt_and_merge(calendar, prices, sales, submission=True)

    # forecast the 28 days for validation
    for day in tqdm(range(0, 28)):
        forecast_date = FIRST_DATE + timedelta(days=day)
        forecast_df = next_day_features(df, forecast_date)

        drop_cols = ['id', 'date', 'sales', 'd', 'wm_yr_wk', 'weekday']
        keep_cols = forecast_df.columns[~forecast_df.columns.isin(drop_cols)]

        forecast_df = forecast_df.loc[forecast_df['date'] == forecast_date, keep_cols]
        df.loc[df['date'] == forecast_date, 'sales'] = model.predict(forecast_df)

    # create the submission file
    submission = make_submission(df)
    submission.to_csv('submission.csv', index=False)

    return submission


if __name__ == "__main__":
    print("Loading training data...")
    start_time = datetime.now()
    calendar, prices, sales = data_frames('../kaggle/input/m5-forecasting-accuracy/')
    calendar, prices, sales = optimize_df(calendar, prices, sales, days=365)
    print("Data load time:", datetime.now() - start_time)

    # Load trained model
    filename = '../models/model_365d_500it.lgb'
    lgb_model = lgb.Booster(model_file=filename)

    submission = infer(lgb_model, calendar, prices, sales)
