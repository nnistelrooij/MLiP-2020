import torch
from tqdm import tqdm


def validate(model, val_loader, val_writer, criterion):
    """Computes loss and score of current state of model on validation data set.

    Args:
        model      = [nn.Module] model to test with validation data set
        val_loader = [DataLoader] validation data loader
        val_writer = [MetricWriter] TensorBoard writer of validation metrics
        criterion  = [nn.Module] neural network module to compute loss
    """
    # set model mode to evaluation
    model.eval()

    with torch.no_grad():
        for data in val_loader:
            x, t_graph, t_vowel, t_conso, _ = data

            # predict
            y = model(x)

            # loss
            t = t_graph, t_vowel, t_conso
            losses = criterion(y, t)

            # accumulate but do not show validation metrics
            val_writer.show_metrics(losses, y, t, inc=False, eval_freq=-1)

    # show validation metrics on TensorBoard
    val_writer.show_metrics(end=True)

    # set model mode back to training
    model.train()


def train(model, train_dataset, train_loader, train_writer,
          val_loader, val_writer, optimizer, criterion, num_epochs=50):
    """Trains the model given train data and validates it given validation data.

    Args:
        model         = [nn.Module] model to train and validate
        train_dataset = [BengaliDataset] train data set
        train_loader  = [DataLoader] train data loader
        train_writer  = [SummaryWriter] TensorBoard writer of train metrics
        val_loader    = [DataLoader] validation data loader
        val_writer    = [SummaryWriter] TensorBoard writer of validation metrics
        optimizer     = [Optimizer] optimizer to update the model
        criterion     = [nn.Module] neural network module to compute loss
        num_epochs    = [int] number of iterations over the train data set
    """
    for epoch in range(num_epochs):
        for data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
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
            train_writer.show_metrics(losses, y, t)

        # evaluate model on validation data
        validate(model, val_loader, val_writer, criterion)

        # reset dataset to keep class balance
        train_dataset.reset(epoch)
