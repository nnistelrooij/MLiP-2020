from datetime import datetime

import torch
import argparse
from torch.optim import Adam
from torchsummary import summary  # pip install torchsummary

from nn import CrossEntropySumLoss, ZeroNet
from optim import train
from utils.data import load_data
from utils.tensorboard import MetricsWriter

# IMAGES = r'kaggle\input\bengaliai-cv19\train_image_data_64.npy'
# LABELS = r'kaggle\input\bengaliai-cv19\train.csv'
# TEST_RATIO = 0.2
# NUM_EPOCHS = 50
# DATA_AUGMENTATION = True
# DROP_INFO_FUNCTION = 'gridmask'  # 'cutout' or None
# CLASS_BALANCING = True
# BATCH_SIZE = 32
IMAGE_SIZE = 64
MODEL = 'model.pt'

if __name__ == '__main__':
    """
    handles input arguments
    main --help gives overview
    """
    drop_info_func = ['cutout', 'gridmask', 'None']
    # process the command options
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', '-i', type=str,
                        help=r'provide path in style: kaggle\input\bengaliai-cv19\train_image_data_64.npy')
    parser.add_argument('--labels', '-l', type=str,
                        help=r'provide path in style: kaggle\input\bengaliai-cv19\train.csv')
    parser.add_argument('--test_ratio', '-tr', type=float, default=0.2,
                        help='[float] percentage of data used for validation, default tr=0.2')
    parser.add_argument('--num_epochs', '-e', type=int, default=50,
                        help='[int] number of iterations over the train data set, default e=50')
    parser.add_argument('--data_augmentation', '-da', type=bool, default=False,
                        help='[bool] whether or not the images are transformed, default da=False')
    parser.add_argument('--drop_info_function', '-drop', type=str, choices=drop_info_func,
                        help=r'[str] whether to use cutout, gridmask or no info dropping algorithm i.e. None')
    parser.add_argument('--class_balancing', '-cb', type=bool, default=False,
                        help='[bool] whether or not the classes are balanced')
    parser.add_argument('--batch_size', '-bs', type=int, default=32,
                        help='[int] batch size of the DataLoader objects default bs=32')
    parser.add_argument('--image_size', '-is', type=int,
                        help='[int] image pixel size of the input image, assumed square image')
    parser.add_argument('--model', '-m', type=str,
                        help='[str]  model name such as model.pt')
    args = parser.parse_args()
    # load training and validation data

    # print('IMAGES = {}'.format(args.images))
    # print('LABELS = {}'.format(args.labels))
    # print('TEST_RATIO = {}'.format(args.test_ratio))
    # print('NUM_EPOCHS = {}'.format(args.num_epochs))
    # print('DATA_AUGMENTATION = {}'.format(args.data_augmentation))
    # print('DROP INFO FUNC = {}'.format(args.drop_info_function))
    # print('CLASS BALANCING = {}'.format(args.class_balancing))
    # print('BATCH SIZE = {}'.format(args.batch_size))
    data = load_data(args.images,
                     args.labels,
                     args.test_ratio,
                     args.num_epochs,
                     args.data_augmentation,
                     args.drop_info_function,
                     args.class_balancing,
                     args.batch_size)
    train_dataset, train_loader, val_loader = data

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # initialize network and show summary
    model = ZeroNet(device).train()
    summary(model, input_size=(1, args.image_size, args.image_size), device=str(device))

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
