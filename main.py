from datetime import datetime

import torch
from torch.optim import Adam
from torchsummary import summary  # pip install torchsummary

from nn import CrossEntropySumLoss, ZeroNet
from optim import train
from utils.data import load_data
from utils.tensorboard import MetricsWriter

IMAGES = r'kaggle\input\bengaliai-cv19\train_image_data_64.npy'
LABELS = r'kaggle\input\bengaliai-cv19\train.csv'
TEST_RATIO = 0.2
NUM_EPOCHS = 50
DATA_AUGMENTATION = True
DROP_INFO_FUNCTION = 'gridmask'  # 'cutout' or None
CLASS_BALANCING = True
BATCH_SIZE = 32
IMAGE_SIZE = 64
MODEL = 'model.pt'

if __name__ == '__main__':
    # load training and validation data
    data = load_data(IMAGES, LABELS, TEST_RATIO, NUM_EPOCHS, DATA_AUGMENTATION,
                     DROP_INFO_FUNCTION, CLASS_BALANCING, BATCH_SIZE)
    train_dataset, train_loader, val_loader = data

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # initialize network and show summary
    model = ZeroNet(device).train()
    summary(model, input_size=(1, IMAGE_SIZE, IMAGE_SIZE), device=str(device))

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
    torch.save(model.state_dict(), MODEL)
