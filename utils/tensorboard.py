import numpy as np
from sklearn.metrics import recall_score
import torch
from torch.utils.tensorboard import SummaryWriter


class MetricsWriter(SummaryWriter):
    """Class to periodically show metrics on TensorBoard.

    Attributes:
        num_images     = [int] number of images the model has been trained on
        num_batches    = [int] number of batches since the metrics were shown
        running_losses = [torch.Tensor] subproblems and combined running losses
        pred_dict      = [dict] dictionary with all predictions from the nn
        true_dict      = [dict] dictionary with all the true labels
    """
    num_images = 0

    def __init__(self, device, log_dir=None):
        """Initialize this class as subclass of SummaryWriter.

        Args:
            device  = [torch.device] device to compute the loss on
            log_dir = [str] directory to store the metrics during the run
        """
        super(MetricsWriter, self).__init__(log_dir)

        self.running_losses = torch.zeros(4, device=device)
        self._reset()

    def _reset(self):
        """Resets the running variables."""
        self.num_batches = 0
        self.running_losses *= 0
        self.true_dict = {'grapheme': [], 'vowel': [], 'consonant': []}
        self.pred_dict = {'grapheme': [], 'vowel': [], 'consonant': []}

    def _eval_metric(self):
        """Competition evaluation metric.

        Adapted from:
        https://www.kaggle.com/c/bengaliai-cv19/overview/evaluation. The metric
        describes the weighted average of component macro-averaged recalls.

        Returns [float]*4:
            grapheme  = grapheme_root component macro-average recall
            vowel     = vowel_diacritic component macro-average recall
            consonant = consonant_diacritic component macro-average recall
            total     = weighted average of component macro-averaged recalls
        """
        scores = []
        for key in ['grapheme', 'vowel', 'consonant']:
            t = self.true_dict[key]
            y = self.pred_dict[key]
            scores.append(recall_score(t, y, average='macro'))

        scores.append(np.average(scores, weights=[2, 1, 1]))
        return scores

    def _update_dicts(self, y, t):
        """Updates two dictionaries given batches of values.

        Args:
            y = [tuple] sequence of tensors of (raw) predictions
            t = [tuple] sequence of tensors of targets
        """
        for key, y, t in zip(['grapheme', 'vowel', 'consonant'], y, t):
            self.true_dict[key] += t.tolist()
            self.pred_dict[key] += y.argmax(dim=1).tolist()

    def show_metrics(self, losses=None, preds=None, targets=None,
                     num_images=0, eval_freq=100000, end=False):
        """Show the losses and scores on TensorBoard.

        Args:
            losses     = [torch.Tensor] subproblem losses and combined loss
            preds      = [tuple] sequence of tensors of (raw) predictions
            targets    = [tuple] sequence of tensors of targets
            num_images = [int] number of unique images in current batch
            eval_freq  = [int] number of images before the next TensorBoard
                               update; if set to -1, TensorBoard never updates
            end        = [bool] always shows metrics after epoch has ended
        """
        if not end:
            # increment total number of training images during run
            MetricsWriter.num_images += num_images

            # increment number of batches to show metrics over
            self.num_batches += 1

            # accumulate metrics to smooth plots
            self.running_losses += losses.data
            self._update_dicts(preds, targets)

        # show metrics every eval_freq images or at the end of an epoch
        if self.num_images % eval_freq == (eval_freq - 1) or end:
            # show losses in TensorBoard
            losses = self.running_losses / self.num_batches
            self.add_scalar('Loss/grapheme_root',
                            losses[0], self.num_images)
            self.add_scalar('Loss/vowel_diacritic',
                            losses[1], self.num_images)
            self.add_scalar('Loss/consonant_diacritic',
                            losses[2], self.num_images)
            self.add_scalar('Loss/total',
                            losses[3], self.num_images)

            # show scores in TensorBoard
            scores = self._eval_metric()
            self.add_scalar('Score/grapheme_root',
                            scores[0], self.num_images)
            self.add_scalar('Score/vowel_diacritic',
                            scores[1], self.num_images)
            self.add_scalar('Score/consonant_diacritic',
                            scores[2], self.num_images)
            self.add_scalar('Score/total',
                            scores[3], self.num_images)

            # reset running variables
            self._reset()
