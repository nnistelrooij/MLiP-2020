import torch
from tqdm import tqdm


class ReduceLROnPlateau(object):
    """Reduce learning rates when metrics have stopped improving.

    Attributes:
        optimizer      = [Optimizer] optimizer containing the learning rates
        factor         = [float] factor by which the learning rates are reduced
        patience       = [int] number of epochs with no improvement after
            which the learning rates are reduced
        best_metrics   = [list] best metrics observed thus far
        num_bad_epochs = number of consecutive epochs with worse metrics
    """

    def __init__(self, optimizer, factor=0.1, patience=4):
        """Initialize the learning rate scheduler.

        Args:
            optimizer = [Optimizer] optimizer containing the learning rates
            factor    = [float] factor by which the learning rate is reduced
            patience  = [int] number of epochs with no improvement after
                which the learning rates are reduced
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.best_metrics = [0]*4
        self.num_bad_epochs = [0]*4

    def step(self, writer, metrics):
        """Update the learning rates for each parameter group given the metrics.

        Args:
            writer  = [MetricWriter] TensorBoard writer of learning rates
            metrics = [float]*4 sub-problem and total scores on validation data

        Returns [bool]:
            Whether the current model has achieved the highest total score.
        """
        for i in range(len(metrics)):
            if self.best_metrics[i] < metrics[i]:
                self.best_metrics[i] = metrics[i]
                self.num_bad_epochs[i] = 0
            else:
                self.num_bad_epochs[i] += 1

            # update learning rate of parameter group responsible for metric
            if self.num_bad_epochs[i] > self.patience:
                self.optimizer.param_groups[i]['lr'] *= self.factor
                self.num_bad_epochs[i] = 0

        # show learning rates on TensorBoard
        learning_rates = [pg['lr'] for pg in self.optimizer.param_groups]
        writer.show_learning_rates(learning_rates)

        return self.best_metrics[-1] == metrics[-1]


def train(model, train_loader, train_writer, optimizer, criterion, epoch):
    """Update model weights given losses on train data batches.

    Args:
        model        = [nn.Module] model to train with train data set
        train_loader = [DataLoader] train data loader
        train_writer = [MetricWriter] TensorBoard writer of train metrics
        optimizer    = [Optimizer] optimizer to update the model
        criterion    = [nn.Module] neural network module to compute losses
        epoch        = [int] current iteration over the train data set
    """
    for data in tqdm(train_loader, desc=f'Train Epoch {epoch}'):
        x, t_graph, t_vowel, t_conso, num_augments = data

        # predict
        y = model(x, num_augments)

        # loss
        t = t_graph, t_vowel, t_conso
        losses = criterion(y, t)

        # update
        optimizer.zero_grad()
        losses[-1].backward()
        optimizer.step()

        # show train metrics every 100 iterations in TensorBoard
        train_writer.show_metrics(y, t, losses, len(x))


def validate(model, val_loader, val_writer, criterion, epoch):
    """Computes losses and scores of current model on validation data.

    Args:
        model      = [nn.Module] model to test with validation data set
        val_loader = [DataLoader] validation data loader
        val_writer = [MetricWriter] TensorBoard writer of validation metrics
        criterion  = [nn.Module] neural network module to compute losses
        epoch      = [int] current iteration over the validation data set

    Returns [float]:
        Sub-problem and total scores on the validation data set.
    """
    # set model mode to evaluation
    model.eval()

    with torch.no_grad():
        for data in tqdm(val_loader, desc=f'Validation Epoch {epoch}'):
            x, t_graph, t_vowel, t_conso, _ = data

            # predict
            y = model(x)

            # loss
            t = t_graph, t_vowel, t_conso
            losses = criterion(y, t)

            # accumulate but do not show validation metrics
            val_writer.show_metrics(y, t, losses, eval_freq=-1)

    # set model mode back to training
    model.train()

    # show validation metrics on TensorBoard
    val_scores = val_writer.show_metrics(end=True)
    return val_scores


def optimize(model,
             train_dataset, train_loader, train_writer,
             val_loader, val_writer,
             optimizer, scheduler,
             criterion,
             num_epochs,
             model_path):
    """Trains and validates model and saves best-performing model.

    Args:
        model         = [nn.Module] model to train and validate
        train_dataset = [BengaliDataset] train data set
        train_loader  = [DataLoader] train data loader
        train_writer  = [MetricWriter] TensorBoard writer of train metrics
        val_loader    = [DataLoader] validation data loader
        val_writer    = [SummaryWriter] TensorBoard writer of validation metrics
        optimizer     = [Optimizer] optimizer to update the model
        scheduler     = [object] scheduler to update the learning rate
        criterion     = [nn.Module] neural network module to compute losses
        num_epochs    = [int] number of iterations over the train data set
        model_path    = [str] path where trained model is saved
    """
    for epoch in range(1, num_epochs + 1):
        # reset dataset for class balancing and to update augment probability
        train_dataset.reset(epoch, num_epochs)

        # update model weights given losses on train data
        train(model, train_loader, train_writer, optimizer, criterion, epoch)

        # determine total score of current model on validation data
        val_scores = validate(model, val_loader, val_writer, criterion, epoch)

        # update learning rates given validation scores
        best_model = scheduler.step(val_writer, val_scores)
        if best_model:  # save best-performing model to storage
            torch.save(model.state_dict(), model_path)
