import math

import torch
from tqdm import tqdm


class ReduceLROnPlateau(object):
    """Reduce learning rate when loss has stopped improving.

    Attributes:
        writer         = [TensorBoardWriter] TensorBoard writer of learning rate
        optimizer      = [Optimizer] optimizer containing the learning rate
        factor         = [float] factor by which the learning rate is reduced
        patience       = [int] number of epochs with no improvement after
            which the learning rate is reduced
        best           = [float] best loss observed thus far
        num_bad_epochs = [int] number of consecutive epochs with a worse loss
    """

    def __init__(self, writer, optimizer, factor=0.1, patience=5):
        """Initialize the learning rate scheduler.

        Args:
            writer    = [TensorBoardWriter] TensorBoard writer of learning rate
            optimizer = [Optimizer] optimizer containing the learning rate
            factor    = [float] factor by which the learning rate is reduced
            patience  = [int] number of epochs with no improvement after
                which the learning rate is reduced
        """
        self.writer = writer
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.best = math.inf
        self.num_bad_epochs = 0

    def step(self, loss):
        """Update the learning rate given the validation loss.

        Args:
            loss = [torch.Tensor] validation loss over one epoch
        """
        if self.best > loss:
            self.best = loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # decrease learning rate after a consistently worse loss
        if self.num_bad_epochs > self.patience:
            params = self.optimizer.param_groups[0]
            params['lr'] *= self.factor
            self.num_bad_epochs = 0

        # show learning rate on TensorBoard
        params = self.optimizer.param_groups[0]
        self.writer.show_learning_rate(params['lr'])


def train(model, train_loader, train_writer, optimizer, criterion, epoch):
    """Update model parameters given losses on train data.

    Args:
        model        = [nn.Module] model to train with train data set
        train_loader = [DataLoader] train data loader
        train_writer = [TensorBoardWriter] TensorBoard writer of train loss
        optimizer    = [Optimizer] optimizer to update the model
        criterion    = [nn.Module] neural network module to compute loss
        epoch        = [int] current iteration over the training data set
    """
    for day, items, t in tqdm(train_loader, desc=f'Train Epoch {epoch}'):
        # predict
        y = model(day, items)

        # compute loss and show on TensorBoard every 100 days
        loss = criterion(y, t)
        train_writer.show_loss(loss, day.shape[1])

        # update model's parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(model, val_loader, val_writer, criterion, epoch):
    """Computes loss of current model on validation data.

    Args:
        model      = [nn.Module] model to test with validation data set
        val_loader = [DataLoader] validation data loader
        val_writer = [TensorBoardWriter] TensorBoard writer of validation loss
        criterion  = [nn.Module] neural network module to compute loss
        epoch      = [int] current iteration over the training data set

    Returns [torch.Tensor]:
        Validation loss over one epoch.
    """
    # set model mode to evaluation
    model.eval()

    # start iterator with actual sales from previous day
    val_loader = iter(val_loader)
    day, items, t = next(val_loader)

    with torch.no_grad():
        # initialize sales and targets columns for current day
        y = model(day, items)
        sales = y[:, :1]
        targets = t[..., :1]

        for day, items, t in tqdm(val_loader, desc=f'Validation Epoch {epoch}'):
            # replace actual sales in items with projected sales
            items[0, 0, 2] = sales[:, -1]

            # predict with sales projections from previous days
            y = model(day, items)

            # add sales projections and targets to tables
            sales = torch.cat((sales, y[:, :1]), dim=1)
            targets = torch.cat((targets, t[..., :1]), dim=2)

    # compute loss over whole horizon and show on TensorBoard
    loss = criterion(sales, targets)
    val_writer.show_loss(loss)

    # set model mode back to training
    model.train()

    return loss


def optimize(model,
             train_loader, train_writer,
             val_loader, val_writer,
             optimizer, scheduler,
             criterion,
             num_epochs,
             model_path):
    """Trains and validates model and saves best-performing model.

    Args:
        model        = [nn.Module] model to train and validate
        train_loader = [DataLoader] DataLoader for training data
        train_writer = [MetricWriter] TensorBoard writer of train metrics
        val_loader   = [DataLoader] DataLoader for validation data
        val_writer   = [MetricWriter] TensorBoard writer of validation metrics
        optimizer    = [Optimizer] optimizer to update the model
        scheduler    = [object] scheduler to update the learning rates
        criterion    = [nn.Module] neural network module to compute losses
        num_epochs   = [int] number of iterations over the training data
        model_path   = [str] path where trained model is saved
    """
    for epoch in range(1, num_epochs + 1):
        # reset hidden state of model each epoch
        model.reset_hidden()
        
        # update model weights given losses on train data
        train(model, train_loader, train_writer, optimizer, criterion, epoch)

        # determine score on validation data
        val_score = validate(model, val_loader, val_writer, criterion, epoch)

        # update learning rate given validation score
        scheduler.step(val_score)

        # save best-performing model to storage
        if val_score == scheduler.best:
            torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    import os
    import pandas as pd
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from nn import WRMSSE
    from utils.data import ForecastDataset
    from utils.tensorboard import TensorBoardWriter

    class Model(nn.Module):
        def __init__(self, num_groups=30490, horizon=5):
            super(Model, self).__init__()

            self.param = nn.Parameter(torch.tensor(10.0))

            self.num_groups = num_groups
            self.horizon = horizon

        def forward(self, x, y):
            return torch.randn(self.num_groups, self.horizon) + self.param


    path = ('D:\\Users\\Niels-laptop\\Documents\\2019-2020\\Machine Learning in'
            ' Practice\\Competition 2\\project\\')
    calendar = pd.read_csv(path + 'calendar.csv')
    prices = pd.read_csv(path + 'sell_prices.csv')
    sales = pd.read_csv(path + 'sales_train_validation.csv')
    train_sales = pd.concat((sales.iloc[:, :6], sales.iloc[:, 1548:-28]), axis=1)
    val_sales = pd.concat((sales.iloc[:, :6], sales.iloc[:, -28:]), axis=1)

    seq_len = 8
    horizon = 5
    train_dataset = ForecastDataset(calendar, prices, train_sales, seq_len, horizon)
    train_loader = DataLoader(train_dataset)

    val_dataset = ForecastDataset(calendar, prices, val_sales)
    val_loader = DataLoader(val_dataset)

    train_writer = TensorBoardWriter(log_dir='runs/train/')
    val_writer = TensorBoardWriter(eval_freq=1, log_dir='runs/val/')

    model = Model(horizon=horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = WRMSSE('cpu', calendar, prices, sales)

    if not os.path.exists('models/'):
        os.mkdir('models/')
    optimize(model, train_loader, train_writer, val_loader, val_writer,
             optimizer, ReduceLROnPlateau(val_writer, optimizer), criterion,
             100, 'models/model.pt')


    i = 3

