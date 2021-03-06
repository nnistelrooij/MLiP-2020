import math

import torch
from tqdm import tqdm

from utils.data import ForecastDataset


class ReduceLROnPlateau(object):
    """Reduce learning rate when loss has stopped improving.

    Attributes:
        writer         = [MetricWriter] TensorBoard writer of learning rate
        optimizer      = [Optimizer] optimizer containing the learning rate
        factor         = [float] factor by which the learning rate is reduced
        patience       = [int] number of epochs with no improvement after
            which the learning rate is reduced
        best           = [float] best loss observed thus far
        num_bad_epochs = [int] number of consecutive epochs with a worse loss
    """

    def __init__(self, writer, optimizer, factor=0.1, patience=10):
        """Initialize the learning rate scheduler.

        Args:
            writer    = [MetricWriter] TensorBoard writer of learning rate
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
        train_writer = [MetricWriter] TensorBoard writer of train loss
        optimizer    = [Optimizer] optimizer to update the model
        criterion    = [nn.Module] neural network module to compute loss
        epoch        = [int] current iteration over the training data set

    Returns [int]:
        Total number of input days given to model during this epoch.
    """
    num_input_days = 0
    for data in tqdm(train_loader, desc=f'Train Epoch {epoch}'):
        day, t_day, items, t_items = data

        # predict sales projections
        y = model(day, t_day[:, :-1], items, t_items[:, :-1])

        # compute loss and show on TensorBoard every eval_freq iterations
        loss = criterion(y, t_items[0, 1:, :, 2])
        train_writer.show_loss(loss, t_day.shape[1] - 1)

        # update model's parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # bookkeeping
        num_input_days += day.shape[1]

    return num_input_days


def validate(model, val_loader, val_writer, criterion,
             epoch, num_input_days, num_val_days):
    """Computes loss of current model on validation data.

    Args:
        model          = [nn.Module] model to test with validation data set
        val_loader     = [DataLoader] validation data loader
        val_writer     = [MetricWriter] TensorBoard writer of validation loss
        criterion      = [nn.Module] neural network module to compute loss
        epoch          = [int] current iteration over the training data set
        num_input_days = [int] total number of input days given to model
        num_val_days   = [int] number of days to use for validation data

    Returns [torch.Tensor]:
        Validation loss over one epoch.
    """
    # set model mode to evaluation
    model.eval()

    # make tqdm iterator and go to current day
    current_day = ForecastDataset.start_idx + num_input_days
    val_iter = iter(tqdm(val_loader, desc=f'Validation Epoch {epoch}'))
    for _ in range(current_day):
        next(val_iter)

    # pass data of last target days through model
    num_target_days = len(val_loader) - num_val_days - current_day
    with torch.no_grad():
        for _ in range(num_target_days):
            day, t_day, items, t_items = next(val_iter)
            model(day, t_day[:, :-1], items, t_items[:, :-1])

    # initialize sales and targets tables to compute loss once
    sales = torch.empty(0, 30490).to(model.device)
    targets = torch.empty(0, 30490)

    with torch.no_grad():
        for day, t_day, items, t_items in val_iter:
            # replace actual sales in items and t_items with projected sales
            items[0, 0:len(sales) > 1, :, 2] = sales[-2:-1]
            t_items[0, 0:len(sales) > 0, :, 2] = sales[-1:]

            # predict with sales projections from previous days
            y = model(day, t_day[:, :-1], items, t_items[:, :-1])

            # add sales projections and targets to tables
            sales = torch.cat((sales, y))
            targets = torch.cat((targets, t_items[0, 1:, :, 2]))

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
             num_val_days,
             model_path):
    """Trains and validates model and saves best-performing model.

    Args:
        model        = [nn.Module] model to train and validate
        train_loader = [DataLoader] DataLoader for training data
        train_writer = [MetricWriter] TensorBoard writer of train metrics
        val_loader   = [DataLoader] DataLoader for validation data
        val_writer   = [MetricWriter] TensorBoard writer of validation metrics
        optimizer    = [Optimizer] optimizer to update the model
        scheduler    = [object] scheduler to update the learning rate
        criterion    = [nn.Module] neural network module to compute losses
        num_epochs   = [int] number of iterations over the training data
        num_val_days = [int] number of days to use for validation data
        model_path   = [str] path where trained model is saved
    """
    for epoch in range(1, num_epochs + 1):
        # reset hidden state of model each epoch
        model.reset_hidden()
        
        # update model weights given losses on train days
        num_input_days = train(
            model, train_loader, train_writer, optimizer, criterion, epoch
        )

        # determine score on validation days
        val_score = validate(
            model, val_loader, val_writer, criterion,
            epoch, num_input_days, num_val_days
        )

        # update learning rate given validation score
        scheduler.step(val_score)

        # save best-performing model to storage
        if val_score == scheduler.best:
            torch.save(model.state_dict(), model_path)
