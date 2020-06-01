import pandas as pd
import torch

from utils.data import data_frames
from nn import WRMSSE


def submission_loss(criterion, submission, sales, start_day, end_day):
    """Computes the WRMSSE of the submission.

    Args:
        criterion  = [WRMSSE] loss function
        submission = [pd.DataFrame] submission in a Pandas DataFrame
        sales      = [pd.DataFrame] actual sales in a Pandas DataFrame
        start_day  = [int] first day of sales of the submission
        end_day    = [int] last day of sales of the submission

    Returns [torch.Tensor]:
        Loss of the submission from start_day up to and including end_day.
    """
    # get sorted projected sales from submission DataFrame
    submission['store_id'] = submission.loc[:, 'id'].map(lambda x: x[-15:-11])
    submission['item_id'] = submission.loc[:, 'id'].map(lambda x: x[:-16])
    submission = submission.sort_values(by=['store_id', 'item_id'])
    submission.index = range(submission.shape[0])
    projected_sales = submission.filter(like='F').to_numpy()
    projected_sales = torch.tensor(projected_sales, dtype=torch.float32)

    # get sorted actual sales from sale DataFrame
    sales = sales.sort_values(by=['store_id', 'item_id'])
    sales.index = range(sales.shape[0])
    actual_sales = sales.loc[:, f'd_{start_day}':f'd_{end_day}'].to_numpy()
    actual_sales = torch.tensor(actual_sales, dtype=torch.float32)

    return criterion(projected_sales.T, actual_sales.T)


if __name__ == '__main__':
    start_day = 1886  # 1914
    end_day = 1913  # 1941
    validation = True

    data_path = ('D:/Users/Niels-laptop/Documents/2019-2020/Machine '
                 'Learning in Practice/Competition 2/project')
    calendar, prices, sales = data_frames(data_path)

    criterion = WRMSSE(calendar, prices, sales, torch.device('cpu'))

    sales = pd.read_csv(data_path + '/sales_train_validation.csv')

    submission_path = 'D:/Users/Niels-laptop/Downloads/submission3 (1).csv'
    submission = pd.read_csv(submission_path)
    if validation:
        submission = submission.iloc[:30490]
    else:
        submission = submission.iloc[30490:]

    loss = submission_loss(criterion, submission, sales, start_day, end_day)
    print(f'Loss from day {start_day} to {end_day}: {loss.item()}')
