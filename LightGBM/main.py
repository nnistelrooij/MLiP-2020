import argparse
from datetime import datetime

import model
from data import data_frames, optimize_df, lgb_dataset
from kaggle  import infer


def handle_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='provide path in style: '
                        r'"kaggle/input/m5-accuracy"')
    parser.add_argument('-d', '--days', type=int, default=365,
                        help='total number of days to train on, default: 365')
    parser.add_argument('-v', '--val_days', type=int, default=0,
                        help='number of validation days, default: 0')
    parser.add_argument('-i', '--iters', type=int, default=200,
                        help='number of training iterations, default: 200')
    parser.add_argument('-k', '--kaggle', action='store_true',
                        help='switch to create a kaggle submission')
    parser.add_argument('-s', '--save', type=str, default=None,
                        help='path to save model, default: None')

    # parse and print arguments
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg.upper()}: {getattr(args, arg)}')

    return args


def run(path, days=None, val_days=0, iters=200, kaggle=False, model_path=None):
    """
    Train a LightGBM model, and optionally create a kaggle submission.

    Args:
        path        = [str] path to folder with competition data
        days        = [int] number of days to keep
        iters       = [int] number of training iterations
        kaggle      = [boolean] if True, create a kaggle submission
        model_path  = [str] path to save model

    Returns [(lgb.Booster, pd.DataFrame)]:
        lgb_model = trained LightGBM model
        submission = submission dataframe
    """
    print("Loading training data...")
    start_time = datetime.now()
    calendar, prices, sales = data_frames(path)
    calendar, prices, sales = optimize_df(calendar, prices, sales, days=days, val_days=val_days)
    train_set, val_set = lgb_dataset(calendar, prices, sales)
    print("Data load time:", datetime.now() - start_time)

    start_time = datetime.now()
    lgb_model = model.train(train_set, val_set, num_rounds=iters, save_model=model_path)
    print("Model train time:", datetime.now() - start_time)

    if kaggle:
        submission = infer(lgb_model, calendar, prices, sales)


if __name__ == "__main__":
    args = handle_arguments()
    run(path=args.path, 
        days=args.days, 
        val_days=args.val_days,
        iters=args.iters,
        kaggle=args.kaggle, 
        model_path=args.save)
