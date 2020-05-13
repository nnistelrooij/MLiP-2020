import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardWriter(SummaryWriter):
    """Class to show losses and learning rates on TensorBoard.

    Attributes:
        num_days     = [int] number of days the model has been training for
        num_batches  = [int] number of batches since writer initialization
        eval_freq    = [int] number of days before the next TensorBoard update
        running_loss = [torch.Tensor] running loss for smoothing loss plot
    """
    num_days = 0

    def __init__(self, eval_freq=100, log_dir=None):
        """Initialize this class as subclass of SummaryWriter.

        Args:
            eval_freq = [int] number of days before the next TensorBoard update
            log_dir   = [str] directory to store the run data file
        """
        super(TensorBoardWriter, self).__init__(log_dir)

        self.num_batches = 0
        self.eval_freq = eval_freq
        self.running_loss = torch.tensor(0.0)

    def show_loss(self, loss, num_days=0):
        """Show the loss on TensorBoard.

        Args:
            loss     = [torch.Tensor] loss on the current day of data
            num_days = [int] number of training days in current batch
        """
        # increase day and batch counters
        TensorBoardWriter.num_days += num_days
        self.num_batches += 1

        # update running loss
        self.running_loss += loss.detach().cpu()

        # show loss every eval_freq batches
        if self.num_batches % self.eval_freq == 0:
            # show loss on TensorBoard
            loss = self.running_loss / self.eval_freq
            self.add_scalar('loss', loss, TensorBoardWriter.num_days)

            # reset running loss
            self.running_loss *= 0

    def show_learning_rate(self, lr):
        """Show the learning rate on TensorBoard.

        Args:
            lr = [float] learning rate on the current epoch
        """
        self.add_scalar('learning rate', lr, TensorBoardWriter.num_days)
