from multiprocessing import Pool, cpu_count

from fbprophet import Prophet
import pandas as pd
from tqdm import tqdm


def split_series(df):
    series = []
    for i in range(1, df.shape[1]):
        df_i = data.iloc[:, [0, i]]
        df_i.columns = ['ds', 'y']
        series.append((i, df_i))

    return series


def run_prophet(pair):
    i, df = pair
    model = Prophet(daily_seasonality=False)
    model.fit(df)
    forecast = model.make_future_dataframe(periods=28, include_history=False)
    forecast = model.predict(forecast)
    return i, forecast


def get_predictions(pairs):
    pairs.sort(key=lambda x: x[0])
    preds = [pair[1][['yhat']].T for pair in pairs]

    return pd.concat(preds)


if __name__ == '__main__':
    sales = pd.read_csv('sales_train_validation.csv')
    cal = pd.read_csv('calendar.csv')[['date', 'd']]
    data = (sales.set_index('id')
                .loc[:, 'd_1':]
                .T
                .merge(cal, left_index=True, right_on='d')
                [['date'] + sales['id'].tolist()]
                )
    del cal

    data = split_series(data)

    p = Pool(cpu_count())
    predictions = list(tqdm(p.imap(run_prophet, data), total=len(data)))
    p.close()
    p.join()

    predictions = get_predictions(predictions)
    predictions.index = range(predictions.shape[0])
    predictions.columns = [f'F{i}' for i in range(1, 29)]
    predictions = pd.concat([sales[['id']], predictions], axis=1)
    predictions.to_csv('submission.csv', index=False)
