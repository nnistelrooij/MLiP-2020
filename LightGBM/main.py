import argparse
from datetime import datetime

import model
from data import data_frames, optimize_df, lgb_dataset


def handle_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='provide path in style: '
                        r'"kaggle/input/m5-accuracy"')
    parser.add_argument('-d', '--days', type=int, default=365,
                        help='total number of days to train on, default: 365')
    parser.add_argument('-s', '--save', type=str, default=None,
                        help='path to save model')

    # parse and print arguments
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg.upper()}: {getattr(args, arg)}')

    return args


def run(path, days, model_path=None):
    print("Loading training data...")
    start_time = datetime.now()
    calendar, prices, sales = data_frames(path)
    calendar, prices, sales = optimize_df(calendar, prices, sales, days=days)
    train_set, val_set = lgb_dataset(calendar, prices, sales)
    print("Data load time:", datetime.now() - start_time)

    start_time = datetime.now()
    lgb_model = model.train(train_set, val_set, save_model=model_path)
    print("Model train time:", datetime.now() - start_time)

    return lgb_model


if __name__ == "__main__":
    args = handle_arguments()
    booster = run(path=args.path, days=args.days, model_path=args.save)
