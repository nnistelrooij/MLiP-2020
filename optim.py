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


def train(model, loader, train_writer, optimizer, criterion, epoch, num_days):
    """Update model parameters given losses on train data.

    Args:
        model        = [nn.Module] model to train with train data set
        train_loader = [DataLoader] train data loader
        train_writer = [TensorBoardWriter] TensorBoard writer of train loss
        optimizer    = [Optimizer] optimizer to update the model
        criterion    = [nn.Module] neural network module to compute loss
        epoch        = [int] current iteration over the training data set
        num_days     = [int] number of days to validate
    """
    for _ in tqdm(range(len(loader) - num_days), desc=f'Train Epoch {epoch}'):
        day, items, t = next(loader)

        # predict
        y = model(day, items)

        # loss
        loss = criterion(y, t)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show loss every 100 days on TensorBoard
        train_writer.show_loss(loss)


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
    day, items, t = next(val_loader)

    with torch.no_grad():
        # initialize sales and targets for current day
        sales = [model(day, items)[:, 0]]
        targets = [t[..., 0]]

        for day, items, t in tqdm(val_loader, desc=f'Validation Epoch {epoch}'):
            ##############################################
            # TODO: replace sales data in `items` that is available in `sales`
            # First iteration only one thingy is replaced.
            # Second iteration only two thingies, third three thingies,
            # and so on until iteration seq_len, because then
            # each iteration seq_len thingies are replaced with the
            # latest seq_len thingies in `sales`.
            # Pleas ask questions if you don't understand.
            # This is immensely complex and I suck at explaining it
            ##############################################
            # predict with sales projections from previous day
            y = model(day, items)
            sales.append(y[:, 0])

            # add target to targets list
            targets.append(t[..., 0])

    # compute loss over whole horizon
    sales = torch.stack(sales, dim=1)
    targets = torch.stack(targets, dim=2)
    loss = criterion(sales, targets)

    # set model mode back to training
    model.train()

    # show validation loss on TensorBoard
    val_writer.show_loss(loss)

    return loss


def optimize(model,
             loader, train_writer,
             val_writer, num_days,
             optimizer, scheduler,
             criterion,
             num_epochs,
             model_path):
    """Trains and validates model and saves best-performing model.

    Args:
        model        = [nn.Module] model to train and validate
        loader       = [DataLoader] data loader
        train_writer = [MetricWriter] TensorBoard writer of train metrics
        val_writer   = [MetricWriter] TensorBoard writer of validation metrics
        num_days     = [int] number of days to validate
        optimizer    = [Optimizer] optimizer to update the model
        scheduler    = [object] scheduler to update the learning rates
        criterion    = [nn.Module] neural network module to compute losses
        num_epochs   = [int] number of iterations over the training data
        model_path   = [str] path where trained model is saved
    """
    for epoch in range(1, num_epochs + 1):
        loader2 = iter(loader)

        # update model weights given losses on train data
        train(model, loader2, train_writer, optimizer, criterion, epoch, num_days)

        # determine score on validation data
        val_score = validate(model, loader2, val_writer, criterion, epoch)

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

    seq_len = 8
    horizon = 5

    loader = ForecastDataset(calendar, prices, sales.iloc[:, :100], seq_len=seq_len, horizon=horizon)
    loader = DataLoader(loader)

    train_writer = TensorBoardWriter('cpu', True, log_dir='runs/train/')
    val_writer = TensorBoardWriter('cpu', False, log_dir='runs/val/')

    model = Model(horizon=horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = WRMSSE('cpu', calendar, prices, sales)

    if not os.path.exists('models/'):
        os.mkdir('models/')
    optimize(model, loader, train_writer, val_writer, 28,
             optimizer, ReduceLROnPlateau(val_writer, optimizer), criterion,
             100, 'models/model.pt')


    i = 3

