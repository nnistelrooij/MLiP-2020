from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from functools import reduce

from fbprophet import Prophet
import pandas as pd
from tqdm import tqdm


def split_series(df):
    """Splits DataFrame into one DataFrame for each time series."""
    series = []
    for i in range(1, df.shape[1]):
        df_i = data.iloc[:, [0, i]]
        df_i.columns = ['ds', 'y']
        series.append((i, df_i))

    return series


def run_prophet(pair):
    """Makes Prophet model on given time series and predicts next 56 sales."""
    time = datetime.now()
    i, df = pair
    model = Prophet(daily_seasonality=False)
    model.fit(df)
    train_time = datetime.now() - time

    time = datetime.now()
    forecast = model.make_future_dataframe(periods=56, include_history=False)
    forecast = model.predict(forecast)
    test_time = datetime.now() - time
    return i, forecast, train_time, test_time


def get_predictions(pairs, parallel):
    """Get all predictions and print test and train times."""
    # get all predictions in sorted order
    pairs.sort(key=lambda x: x[0])
    preds = [pair[1][['yhat']] for pair in pairs]
    preds = pd.concat(preds, axis=1)

    # select validation and evaluation predictions from all predictions
    validation = preds[:-28].T
    validation.index = range(validation.shape[0])
    evaluation = preds[-28:].T
    evaluation.index = range(evaluation.shape[0])

    # print parallel train and test times
    train_time = reduce(lambda x, y: x + y, [pair[2] for pair in pairs])
    test_time = reduce(lambda x, y: x + y, [pair[3] for pair in pairs])
    print('Train time:', train_time / (train_time + test_time) * parallel)
    print('Test time:', test_time / (train_time + test_time) * parallel)

    return validation, evaluation


def submit(sales, validation, evaluation, submission_path):
    """Saves predictions to storage in one file."""
    # columns used by Kaggle
    columns = [f'F{i}' for i in range(1, 29)]

    # make validation projections in Kaggle format
    id_col = pd.DataFrame(sales.columns)
    validation.columns = columns
    validation = pd.concat((id_col, validation), axis=1)

    # make evaluation projections in Kaggle format
    eval_id_col = id_col.applymap(lambda x: x[:-10] + 'evaluation')
    evaluation.columns = columns
    evaluation = pd.concat((eval_id_col, evaluation), axis=1)

    # concatenate all projections and save to storage
    projections = pd.concat((validation, evaluation), axis=0)
    projections.to_csv(submission_path, index=False)


if __name__ == '__main__':
    time = datetime.now()

    num_days = 1000
    path = Path(r'D:\Users\Niels-laptop\Documents\2019-2020\Machine Learning '
                r'in Practice\Competition 2\project')
    submission_path = 'submission_prophet.csv'

    sales = pd.read_csv(path / 'sales_train_validation.csv')
    sales = sales.sort_values(by=['store_id', 'item_id'])
    sales = sales.set_index('id')
    sales = sales.iloc[:, -num_days - 28:-28]
    sales = sales.T

    cal = pd.read_csv(path / 'calendar.csv')[['date', 'd']]

    data = sales.merge(cal, left_index=True, right_on='d')
    data = data[['date'] + sales.columns.tolist()]
    data = split_series(data)

    p = Pool(cpu_count())
    print('Preparation time:', datetime.now() - time)
    time = datetime.now()
    predictions = list(tqdm(p.imap(run_prophet, data), total=len(data)))
    p.close()
    p.join()

    parallel = datetime.now() - time
    time = datetime.now()
    validation, evaluation = get_predictions(predictions, parallel)

    submit(sales, validation, evaluation, submission_path)
    print('Submission time:', datetime.now() - time)
