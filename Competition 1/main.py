import argparse
from datetime import datetime

import torch
from torch.optim import Adam
from torchsummary import summary  # pip install torchsummary

from nn import BengaliNet, CrossEntropySumLoss, LabelSmoothingLoss
from optim import optimize, ReduceLROnPlateau
from utils.data import load_data
from utils.tensorboard import MetricWriter


def handle_arguments():
    """Handles console arguments. `python main.py --help` gives an overview.

    Returns [Namespace]:
        images            = [str] path to images .npy file
        labels            = [str] path to labels CSV file
        test_ratio        = [float] proportion of data for testing
        seed              = [int] seed used for data splitting
        data_augmentation = [bool] whether to augment the training images
        drop_info_fn      = [str] which info dropping algorithm to use
        class_balancing   = [bool] whether to perform class balancing
        batch_size        = [int] number of images in a batch
        label_smoothing   = [bool] whether to use soft targets for the loss
        epochs            = [int] number of iterations over the training data
        model             = [str] path to save the trained model
    """
    # process the command options
    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=str, help='provide path in style: '
                        r'"kaggle\input\bengaliai-cv19\images.npy"')
    parser.add_argument('labels', type=str, help='provide path in style: '
                        r'"kaggle\input\bengaliai-cv19\labels.csv"')
    parser.add_argument('-t', '--test_ratio', type=float, default=0.2,
                        help='proportion of data for testing, default: 0.2')
    parser.add_argument('-s', '--seed', type=int, default=None, help='seed '
                        'used for consistent data splitting, default: None')
    parser.add_argument('-a', '--data_augmentation', action='store_true',
                        help='switch to augment the images')
    drop_info_fns = ['cutout', 'gridmask', 'None']  # info dropping algorithms
    parser.add_argument('-d', '--drop_info_fn', type=str, choices=drop_info_fns,
                        default=None, help='whether cutout, GridMask, or no '
                        'information dropping algorithm is used, default: None')
    parser.add_argument('-c', '--class_balancing', action='store_true',
                        help='switch to perform class balancing')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch size of DataLoader objects, default: 32')
    parser.add_argument('-l', '--label_smoothing', action='store_true',
                        help='switch to use soft targets in loss computation')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number '
                        'of iterations over training data, default: 50')
    parser.add_argument('-m', '--model', type=str, default='model.pt',
                        help='path to save trained model, default: "model.pt"')

    # parse and print arguments
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg.upper()}: {getattr(args, arg)}')

    return args
  

if __name__ == '__main__':
    # get console arguments
    args = handle_arguments()

    # load training and validation data
    data = load_data(args.images,
                     args.labels,
                     args.test_ratio, args.seed,
                     args.data_augmentation, args.drop_info_fn,
                     args.class_balancing,
                     args.batch_size)
    train_dataset, train_loader, val_loader, image_size = data

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE:', device)

    # initialize network and show summary
    model = BengaliNet(device).train()
    summary(model, input_size=(1, image_size, image_size), device=str(device))

    # initialize optimizer, scheduler, and criterion
    optimizer = Adam([
        {'params': list(model.parameters())[-6:-4], 'lr': 0.001},
        {'params': list(model.parameters())[-4:-2], 'lr': 0.001},
        {'params': list(model.parameters())[-2:], 'lr': 0.001},
        {'params': list(model.parameters())[:-6], 'lr': 0.001},
    ])
    scheduler = ReduceLROnPlateau(optimizer)
    if args.label_smoothing:
        criterion = LabelSmoothingLoss(device, 0.1)
    else:
        criterion = CrossEntropySumLoss(device)

    # TensorBoard writers
    current_time = datetime.now().strftime("%Y-%m-%d/%H'%M'%S")
    train_writer = MetricWriter(device, f'runs/{current_time}/train')
    train_writer.add_graph(model, next(iter(train_loader))[0])   # show model
    val_writer = MetricWriter(device, f'runs/{current_time}/validation')

    # train and validate model
    optimize(model,
             train_dataset, train_loader, train_writer,
             val_loader, val_writer,
             optimizer, scheduler,
             criterion,
             args.epochs,
             args.model)

    # close TensorBoard writers to flush communication
    train_writer.close()
    val_writer.close()
