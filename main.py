from datetime import datetime

import torch
from torch.optim import Adam
from torchsummary import summary  # pip install torchsummary

from nn import CrossEntropySumLoss, ZeroNet
from optim import train
from utils.data import load_data
from utils.tensorboard import MetricsWriter

SIZE = 64  # length (in pixels) of the images
BATCH_SIZE = 32  # batch size of the data loaders
PATH = 'model.pt'  # where to store the trained network

if __name__ == '__main__':
    # load training and validation data
    data = load_data('train_image_data_64.npy', 'train.csv', SIZE, BATCH_SIZE,
                     augment=False, balance=False)
    train_dataset, train_loader, val_loader = data

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # initialize network and show summary
    model = ZeroNet(device).train()
    summary(model, input_size=(1, SIZE, SIZE), device=str(device))

    # initialize optimizer and criterion
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropySumLoss(device)

    # TensorBoard writers
    current_time = datetime.now().strftime("%Y-%m-%d/%H'%M'%S")
    train_writer = MetricsWriter(device, f'runs/{current_time}/train')
    train_writer.add_graph(model, iter(train_loader).next()[0])   # show model
    val_writer = MetricsWriter(device, f'runs/{current_time}/validation')

    # train and validate model
    train(model, train_dataset, train_loader, train_writer,
          val_loader, val_writer, optimizer, criterion)

    # save model to storage
    torch.save(model.state_dict(), PATH)
