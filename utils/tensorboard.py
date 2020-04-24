import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardWriter(SummaryWriter):
    """Class to show losses and learning rates on TensorBoard."""
    _num_days = 0

    def __init__(self, device, train, eval_freq, log_dir=None):
        """Initialize this class as subclass of SummaryWriter.

        Args:
            device    = [torch.device] device to compute the running loss on
            train     = [bool] whether the losses are for training or validation
            eval_freq = [int] number of days before the next TensorBoard update
            log_dir   = [str] directory to store the run data file
        """
        super(TensorBoardWriter, self).__init__(log_dir)

        self._train = train
        self._eval_freq = eval_freq
        self._running_loss = torch.zeros(1, device=device)

    def show_loss(self, loss):
        """Show the loss on TensorBoard.

        Args:
            loss = [torch.Tensor] loss on the current day of data
        """
        # increment day counter
        TensorBoardWriter._num_days += self._train

        # update running loss
        self._running_loss += loss.data

        # show loss every eval_freq days
        if self._num_days % self._eval_freq == 0:
            # show loss on TensorBoard
            loss = self._running_loss / self._eval_freq
            self.add_scalar('loss', loss, TensorBoardWriter._num_days)

            # reset running loss
            self._running_loss *= 0

    def show_learning_rate(self, lr):
        """Show the learning rate on TensorBoard.

        Args:
            lr = [float] learning rate on the current day of data
        """
        self.add_scalar('learning rate', lr, TensorBoardWriter._num_days)
