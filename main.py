from datetime import datetime

import torch
import argparse
from torch.optim import Adam
from torchsummary import summary  # pip install torchsummary

from nn import CrossEntropySumLoss, ZeroNet
from optim import train
from utils.data import load_data
from utils.tensorboard import MetricsWriter


def handle_arguments():
    """Handles input arguments. `python main.py --help` gives an overview."""
    # options for the information dropping algorithms
    drop_info_fn = ['cutout', 'gridmask', 'none']

    # process the command options
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', '-i', type=str, help=r'provide path in '
                        r'style: "kaggle\input\bengaliai-cv19\images.npy"')
    parser.add_argument('--labels', '-l', type=str, help=r'provide path in '
                        r'style: "kaggle\input\bengaliai-cv19\labels.csv"')
    parser.add_argument('--test_ratio', '-tr', type=float, default=0.2,
                        help='proportion of data for testing, default: 0.2')
    parser.add_argument('--num_epochs', '-e', type=int, default=50,
                        help='number of runs over train data, default: 50')
    parser.add_argument('--data_augmentation', '-da', type=bool, default=False,
                        help='whether the images are augmented, default: False')
    parser.add_argument('--drop_info_fn', '-di', type=str, choices=drop_info_fn,
                        default='gridmask', help='whether cutout, GridMask, or '
                        'no information dropping is used, default: "gridmask"')
    parser.add_argument('--class_balancing', '-cb', type=bool, default=False,
                        help='whether the classes are balanced, default: False')
    parser.add_argument('--batch_size', '-bs', type=int, default=32,
                        help='batch size of DataLoader objects, default: 32')
    parser.add_argument('--image_size', '-is', type=int, default=64,
                        help='length of square images in pixels, default: 64')
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
                     args.test_ratio,
                     args.num_epochs,
                     args.data_augmentation,
                     args.drop_info_fn,
                     args.class_balancing,
                     args.batch_size)
    train_dataset, train_loader, val_loader = data

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # initialize network and show summary
    model = ZeroNet(device).train()
    input_size = 1, args.image_size, args.image_size
    summary(model, input_size=input_size, device=str(device))

    # initialize optimizer and criterion
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropySumLoss(device)

    # TensorBoard writers
    current_time = datetime.now().strftime("%Y-%m-%d/%H'%M'%S")
    train_writer = MetricsWriter(device, f'runs/{current_time}/train')
    train_writer.add_graph(model, next(iter(train_loader))[0])   # show model
    val_writer = MetricsWriter(device, f'runs/{current_time}/validation')

    # train and validate model
    train(model, train_dataset, train_loader, train_writer,
          val_loader, val_writer, optimizer, criterion)

    # save model to storage
    torch.save(model.state_dict(), args.model)
