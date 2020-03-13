import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CrossEntropySumLoss(nn.Module):
    """Neural network module to compute sum of cross-entropy losses.

    Attributes:
        device = [torch.device] device to compute the loss on
    """

    def __init__(self, device):
        """Initializes the loss module.

        Args:
            device = [torch.device] device to compute the loss on
        """
        super(CrossEntropySumLoss, self).__init__()
        self.device = device

    def forward(self, input, target):
        """Sums cross-entropy losses of given predictions and targets.

        Args:
            input  = [tuple] sequence of tensors of (raw) predictions
            target = [tuple] sequence of tensors of targets

        Returns [torch.Tensor]:
            The grapheme_root, vowel_dacritic, consonant_diacritic,
            and combined losses, given the predictions and targets.
        """
        losses = []
        for y, t in zip(input, target):
            t = t.to(self.device)
            loss = F.cross_entropy(y, t)
            losses.append(loss)

        losses.append(sum(losses))
        return torch.stack(losses)

    
class LabelSmoothingLoss(nn.Module):
    """Sum of cross-entropy losses with soft targets.

    When `smoothing=0.0`, the loss will be equivalent to 
    standard cross-entropy loss (`F.cross_entropy()`).

    Attributes:
        device     = [torch.device] device to compute the loss on
        smoothing  = [float] controls degree of smoothing, in range [0, 1)
        confidence = [float] max probability in smoothed labels, 1 - smoothing
    """
    def __init__(self, device, smoothing=0.0):
        """Initializes label smoothing loss.

        Args:
            device    = [torch.device] device to compute the loss on
            smoothing = [float] controls degree of smoothing, in range [0, 1)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.device = device
        self.smoothing = smoothing  # alpha
        self.confidence = 1 - smoothing

    def forward(self, input, target):
        """Sums cross-entropy losses, given predictions and soft targets.

        Args:
            input  = [tuple] sequence of tensors of (raw) predictions
            target = [tuple] sequence of tensors of hard targets
            
        Returns [torch.Tensor]:
            The grapheme_root, vowel_dacritic, consonant_diacritic, and
            combined losses, given the predictions and soft targets.
        """
        losses = []
        for y, t in zip(input, target):
            num_classes = y.size(-1)
            t = t.unsqueeze(-1).to(self.device)

            # compute smoothed labels
            t_smooth = torch.full_like(y, self.smoothing / (num_classes - 1))
            t_smooth = t_smooth.scatter(-1, t, self.confidence)

            # compute smoothed cross-entropy loss
            y = y.log_softmax(dim=-1)
            loss = (-t_smooth * y).sum(dim=-1).mean()
            losses.append(loss)
                
        losses.append(sum(losses))
        return torch.stack(losses)


def _split_vectors(vectors, num_augments):
    """Returns subsets of the latent vectors as tensors for each sub-problem.

    Subsets the latent vectors according to the number of augmentations per
    image for each sub-problem. It returns three tensors that contain a
    subset of the latent vectors in vectors to increase training efficiency.

    Args:
        vectors      = [torch.Tensor] the latent vectors to be subsetted
        num_augments = [torch.Tensor] number of augmentations per
            sub-problem with shape (BATCH_SIZE, 3)

    Returns [torch.Tensor]*3:
        The latent vectors for the grapheme_root, vowel_diacritic,
        and consonant_diacritic sub-problems.
    """
    if num_augments is None:
        return vectors, vectors, vectors

    # determine the ranges of the latent vectors for each sub-problem
    max_augments, _ = num_augments.max(dim=-1, keepdim=True)
    diffs = torch.cat((torch.tensor([[0]]), max_augments))
    start_indices = torch.cumsum(diffs, dim=0)[:-1]
    ranges = torch.cat((start_indices, start_indices + num_augments), dim=-1)

    # determine the indices of the latent vectors for each sub-problem
    graph = torch.cat([torch.arange(st, end) for st, end in ranges[:, [0, 1]]])
    vowel = torch.cat([torch.arange(st, end) for st, end in ranges[:, [0, 2]]])
    conso = torch.cat([torch.arange(st, end) for st, end in ranges[:, [0, 3]]])

    return vectors[graph], vectors[vowel], vectors[conso]


class ZeroNet(nn.Module):
    """Simple convolutional neural network.

    Attributes:
        conv1      = [nn.Module] first convolutional layer
        conv2      = [nn.Module] second convolutional layer
        conv2_drop = [nn.Module] 2D dropout layer
        fc1        = [nn.Module] first fully-connected layer
        fc2        = [nn.Module] second fully-connected layer
        fc3        = [nn.Module] third fully-connected layer
        fc4        = [nn.Module] fourth fully-connected layer
        device     = [torch.device] device to compute the predictions on
    """

    def __init__(self, device, kernel_size=3):
        """Initialize the simple convolutional neural network.

        Args:
            device      = [torch.device] device to compute the predictions on
            kernel_size = [int] kernel size for the two convolutional layers
        """
        super(ZeroNet, self).__init__()

        # images are 128 * 128
        # input channels 1, output channels 10
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)

        # input channels 10, output channels 20
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d()

        # extra fully-connected layers to determine labels
        self.fc1 = nn.Linear(720, 256)
        self.fc2 = nn.Linear(256, 168)
        self.fc3 = nn.Linear(256, 11)
        self.fc4 = nn.Linear(256, 7)

        # put model on GPU
        self.device = device
        self.to(self.device)

    def forward(self, x, num_augments=None):
        """Forward pass of the CNN.

        Args:
            x            = [torch.Tensor] images with shape (N, 1, SIZE, SIZE)
            num_augments = [torch.Tensor] number of augmentations per
                sub-problem with shape (BATCH_SIZE, 3)

        Returns [torch.Tensor]*3:
            Non-normalized predictions for each class for each sub-problem.
        """
        # put images on GPU
        x = x.to(self.device)

        # get smaller and denser representations
        h = F.relu(F.max_pool2d(self.conv1(x), 3))
        h = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(h)), 3))
        h = h.flatten(start_dim=1)
        h = F.relu(self.fc1(h))

        # determine predictions for each sub-problem
        h_graph, h_vowel, h_conso = _split_vectors(h, num_augments)
        y_graph = self.fc2(h_graph)
        y_vowel = self.fc3(h_vowel)
        y_conso = self.fc4(h_conso)
        return y_graph, y_vowel, y_conso


class BengaliNet(nn.Module):
    """Model that uses pre-trained ResNet-18 for intermediate layers.

    Attributes:
        conv1    = [nn.Module] first convolutional layer
        resnet18 = [nn.Module] non-pretrained layers of ResNet-18 architecture
        fc2      = [nn.Module] first fully-connected layer
        fc3      = [nn.Module] second fully-connected layer
        fc4      = [nn.Module] third fully-connected layer
        device   = [torch.device] device to compute the predictions on
    """

    def __init__(self, device):
        super(BengaliNet, self).__init__()

        # convolutional layer to get required number of channels
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2, bias=False)

        # create large non-pretrained ResNet model to generate image embeddings
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[1:-1])

        # extra fully-connected layers to determine labels
        self.fc2 = nn.Linear(512, 168)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 7)

        # put model on GPU
        self.device = device
        self.to(self.device)

    def forward(self, x, num_augments=None):
        """Foward pass of the CNN.

        Args:
            x            = [torch.Tensor] images with shape (N, 1, SIZE, SIZE)
            num_augments = [torch.Tensor] number of augmentations per
                sub-problem with shape (BATCH_SIZE, 3)

        Returns [torch.Tensor]*3:
            Non-normalized predictions for each class for each sub-problem.
        """
        # put images on GPU
        x = x.to(self.device)

        # get correct number of channels for ResNet-18
        h = self.conv1(x)

        # get image embeddings from ResNet-18
        h = self.resnet18(h)
        h = h.flatten(start_dim=1)

        # determine predictions for each sub-problem
        h_graph, h_vowel, h_conso = _split_vectors(h, num_augments)
        y_graph = self.fc2(h_graph)
        y_vowel = self.fc3(h_vowel)
        y_conso = self.fc4(h_conso)
        return y_graph, y_vowel, y_conso
