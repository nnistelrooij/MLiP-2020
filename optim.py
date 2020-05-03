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
    val_loader = iter(val_loader)
    day, items, t = next(val_loader)

    with torch.no_grad():
        # initialize sales and targets for current day
        sales = [model(day, items)[0]]
        targets = [t]

        for day, items, t in tqdm(val_loader, desc=f'Validation Epoch {epoch}'):
            # predict with sales projections from previous day
            y = model(day, items, sales)
            sales.append(y[0])

            # add target to targets list
            targets.append(t)

    # compute loss over whole horizon
    sales = torch.stack(sales, dim=1)
    targets = torch.stack(targets, dim=1)
    loss = criterion(sales, targets)

    # set model mode back to training
    model.train()

    # show validation loss on TensorBoard
    val_writer.show_loss(loss)

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
        train_loader = [DataLoader] train data loader
        train_writer = [MetricWriter] TensorBoard writer of train metrics
        val_loader   = [DataLoader] validation data loader
        val_writer   = [MetricWriter] TensorBoard writer of validation metrics
        optimizer    = [Optimizer] optimizer to update the model
        scheduler    = [object] scheduler to update the learning rates
        criterion    = [nn.Module] neural network module to compute losses
        num_epochs   = [int] number of iterations over the training data
        model_path   = [str] path where trained model is saved
    """
    for epoch in range(1, num_epochs + 1):
        # update model weights given losses on train data
        train(model, train_loader, train_writer, optimizer, criterion, epoch)

        # determine score on validation data
        val_score = validate(model, val_loader, val_writer, criterion, epoch)

        # update learning rate given validation score
        scheduler.step(val_score)

        # save best-performing model to storage
        if val_score == scheduler.best:
            torch.save(model.state_dict(), model_path)
