import argparse
from datetime import datetime

import torch

from nn import Model, WRMSSE
from optim import optimize, ReduceLROnPlateau
from utils.data import data_frames, data_loaders
from utils.tensorboard import MetricWriter


def handle_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,
                        help='provide path where CSV data is stored')
    parser.add_argument('-t', '--num_days', type=int, default=365,
                        help='total number of days to train on, default: 365')
    parser.add_argument('-i', '--seq_len', type=int, default=8,
                        help='sequence length of input to model, default: 8')
    parser.add_argument('-o', '--horizon', type=int, default=5,
                        help='sequence length of output of model, default: 5')
    parser.add_argument('-v', '--num_val_days', type=int, default=28,
                        help='number of days for validation, default: 28')
    parser.add_argument('-s', '--num_models', type=int, default=500,
                        help='number of submodels to make, default: 500')
    parser.add_argument('-d', '--dropout', type=float, default=1.0,
                        help='prob of zero inter-group weight, default: 1.0')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='number of iters over training data, default: 50')
    parser.add_argument('-m', '--model', type=str, default='models/model.pt',
                        help='path to save model, default: "models/model.pt"')

    # parse and print arguments
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg.upper()}: {getattr(args, arg)}')

    return args


if __name__ == '__main__':
    # get console arguments
    args = handle_arguments()

    # load Pandas DataFrames
    calendar, prices, sales = data_frames(args.path)

    # get DataLoaders
    train_loader, val_loader = data_loaders(
        calendar, prices, sales,
        args.num_days, args.num_val_days,
        args.seq_len, args.horizon
    )

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE:', device)

    # initialize network and show summary
    model = Model(args.num_models, device, args.dropout)
    # TensorBoard writers
    current_time = datetime.now().strftime("%Y-%m-%d/%H'%M'%S")
    train_writer = MetricWriter(f'runs/{current_time}/train')
    val_writer = MetricWriter(f'runs/{current_time}/validation', eval_freq=1)

    # initialize optimizer, scheduler, and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(val_writer, optimizer)
    criterion = WRMSSE(calendar, prices, sales, device)

    # train and validate model
    optimize(model,
             train_loader, train_writer,
             val_loader, val_writer,
             optimizer, scheduler,
             criterion,
             args.epochs,
             args.model)

    # close TensorBoard writers to flush communication
    train_writer.close()
    val_writer.close()
