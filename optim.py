import torch
from tqdm import tqdm


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
        Total score on the validation data set.
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
    val_score = val_writer.show_metrics(end=True)
    return val_score


def optimize(model,
             train_dataset, train_loader, train_writer,
             val_loader, val_writer,
             optimizer, scheduler,
             criterion,
             num_epochs):
    """Trains the model given train data and validates it given validation data.

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
    """
    for epoch in range(1, num_epochs + 1):
        # reset dataset for class balancing and to update augment probability
        train_dataset.reset(epoch)

        # update model weights given losses on train data
        train(model, train_loader, train_writer, optimizer, criterion, epoch)

        # determine total score of current model on validation data
        val_score = validate(model, val_loader, val_writer, criterion, epoch)

        # update learning rate given validation score
        scheduler.step(val_score)
