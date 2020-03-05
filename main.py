from datetime import datetime

import torch
import argparse
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary  # pip install torchsummary

from nn import CrossEntropySumLoss, LabelSmoothingLoss
from nn import ZeroNet, BengaliNet
from optim import optimize
from utils.data import load_data
from utils.tensorboard import MetricWriter


def handle_arguments():
    """Handles input arguments. `python main.py --help` gives an overview."""
    # options for the information dropping algorithms
    drop_info_fn = ['cutout', 'gridmask', 'None']

    # process the command options
    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=str, help='provide path in style: '
                        r'"kaggle\input\bengaliai-cv19\images.npy"')
    parser.add_argument('labels', type=str, help='provide path in style: '
                        r'"kaggle\input\bengaliai-cv19\labels.csv"')
    parser.add_argument('--test_ratio', '-tr', type=float, default=0.2,
                        help='proportion of data for testing, default: 0.2')
    parser.add_argument('--seed', '-s', type=int, default=None, help='seed '
                        'used for consistent data splitting, default: None')
    parser.add_argument('--data_augmentation', '-da', type=bool, default=False,
                        help='whether the images are augmented, default: False')
    parser.add_argument('--drop_info_fn', '-di', type=str, choices=drop_info_fn,
                        default=None, help='whether cutout, GridMask, or no '
                        'information dropping algorithm is used, default: None')
    parser.add_argument('--class_balancing', '-cb', type=bool, default=False,
                        help='whether the classes are balanced, default: False')
    parser.add_argument('--batch_size', '-bs', type=int, default=32,
                        help='batch size of DataLoader objects, default: 32')
    parser.add_argument('--label_smoothing', '-ls', type=bool, default=False,
                        help='whether the labels are smoothed, default: False')
    parser.add_argument('--num_epochs', '-e', type=int, default=50,
                        help='number of runs over train data, default: 50')
    parser.add_argument('--model', '-m', type=str, default='model.pt',
                        help='path to save trained model, default: "model.pt"')

    # parse and print arguments
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg.upper()}: {getattr(args, arg)}')

    return args
  

if __name__ == '__main__':
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
    print('Device:', device)

    # initialize network and show summary
    model = BengaliNet(device).train()
    summary(model, input_size=(1, image_size, image_size), device=str(device))

    # initialize optimizer, scheduler, and criterion
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, verbose=True)
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
             args.num_epochs,
             args.model)
