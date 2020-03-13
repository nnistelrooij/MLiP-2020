import numpy as np
from sklearn.metrics import recall_score
import torch
from torch.utils.tensorboard import SummaryWriter


class MetricWriter(SummaryWriter):
    """Class to periodically show metrics on TensorBoard.

    Attributes:
        num_images     = [int] number of images the model has been trained on
        num_batches    = [int] number of batches since the metrics were shown
        pred_dict      = [dict] dictionary with all predictions from the model
        true_dict      = [dict] dictionary with all the true labels
        running_losses = [torch.Tensor] sub-problems and combined running losses
    """
    num_images = 0

    def __init__(self, device, log_dir=None):
        """Initialize this class as subclass of SummaryWriter.

        Args:
            device  = [torch.device] device to compute the running loss on
            log_dir = [str] directory to store the metrics during the run
        """
        super(MetricWriter, self).__init__(log_dir)

        self.running_losses = torch.zeros(4, device=device)
        self._reset()

    def _reset(self):
        """Resets the running variables."""
        self.num_batches = 0
        self.pred_dict = {'grapheme': [], 'vowel': [], 'consonant': []}
        self.true_dict = {'grapheme': [], 'vowel': [], 'consonant': []}
        self.running_losses *= 0

    def _eval_metric(self):
        """Competition evaluation metric.

        Adapted from:
        https://www.kaggle.com/c/bengaliai-cv19/overview/evaluation. The metric
        describes the weighted average of component macro-averaged recalls.

        Returns [float]*4:
            grapheme  = grapheme_root component macro-averaged recall
            vowel     = vowel_diacritic component macro-averaged recall
            consonant = consonant_diacritic component macro-averaged recall
            total     = weighted average of component macro-averaged recalls
        """
        scores = []
        for key in ['grapheme', 'vowel', 'consonant']:
            y = self.pred_dict[key]
            t = self.true_dict[key]
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
            self.pred_dict[key] += y.data.argmax(dim=-1).tolist()
            self.true_dict[key] += t.tolist()

    def _show_scores(self, scores):
        """Show scores in TensorBoard.

        Args:
            scores = [tuple] sub-problem and combined scores of current model
        """
        self.add_scalar('Score/grapheme_root', scores[0], self.num_images)
        self.add_scalar('Score/vowel_diacritic', scores[1], self.num_images)
        self.add_scalar('Score/consonant_diacritic', scores[2], self.num_images)
        self.add_scalar('Score/total', scores[3], self.num_images)

    def _show_losses(self, losses):
        """Show losses in TensorBoard.

        Args:
            losses = [torch.Tensor] sub-problem losses and combined loss
        """
        self.add_scalar('Loss/grapheme_root', losses[0], self.num_images)
        self.add_scalar('Loss/vowel_diacritic', losses[1], self.num_images)
        self.add_scalar('Loss/consonant_diacritic', losses[2], self.num_images)
        self.add_scalar('Loss/total', losses[3], self.num_images)

    def show_metrics(self, preds=None, targets=None, losses=None,
                     num_images=0, eval_freq=100, end=False):
        """Show the losses and scores on TensorBoard.

        Args:
            preds      = [tuple] sequence of tensors of (raw) predictions
            targets    = [tuple] sequence of tensors of targets
            losses     = [torch.Tensor] sub-problem and combined losses
            num_images = [int] number of unique images in current train batch
            eval_freq  = [int] number of batches before the next TensorBoard
                update; if set to -1, TensorBoard never updates
            end        = [bool] always shows metrics after epoch has ended

        Returns [float]*4:
            grapheme  = grapheme_root component macro-averaged recall
            vowel     = vowel_diacritic component macro-averaged recall
            consonant = consonant_diacritic component macro-averaged recall
            total     = weighted average of component macro-averaged recalls
        """
        if not end:
            # increment total number of training images during run
            MetricWriter.num_images += num_images

            # increment number of batches to show metrics over
            self.num_batches += 1

            # accumulate metrics to smooth plots
            self._update_dicts(preds, targets)
            self.running_losses += losses.data

        # show metrics every eval_freq batches or at the end of an epoch
        if self.num_batches == eval_freq or end:
            # show scores on TensorBoard
            scores = self._eval_metric()
            self._show_scores(scores)

            # show losses on TensorBoard
            losses = self.running_losses / self.num_batches
            self._show_losses(losses)

            # reset running variables
            self._reset()

            return scores

    def show_learning_rates(self, learning_rates):
        """Show learning rates in TensorBoard.

        Args:
            learning_rates = [float]*4 sub-problem and total learning rates
        """
        self.add_scalar('Learning Rate/grapheme_root',
                        learning_rates[0], self.num_images)
        self.add_scalar('Learning Rate/vowel_diacritic',
                        learning_rates[1], self.num_images)
        self.add_scalar('Learning Rate/consonant_diacritic',
                        learning_rates[2], self.num_images)
        self.add_scalar('Learning Rate/total',
                        learning_rates[3], self.num_images)
