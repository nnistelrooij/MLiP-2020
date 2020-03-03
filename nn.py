import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CrossEntropySumLoss(nn.Module):
    """Neural network module to compute sum of cross entropy losses.

    Attributes:
        device = [torch.device] device to compute the loss on
    """

    def __init__(self, device):
        """Initializes the loss module

        Args:
            device = [torch.device] device to compute the loss on
        """
        super(CrossEntropySumLoss, self).__init__()
        self.device = device

    def forward(self, input, target):
        """Sums cross entropy losses of given predictions and targets.

        Args:
            input  = [tuple] sequence of tensors of (raw) predictions
            target = [tuple] sequence of tensors of targets

        Returns [torch.Tensor]:
            The grapheme_root, vowel_dacritic, consonant_diacritic,
            and combined losses given the predictions and targets.
        """
        losses = []
        for y, t in zip(input, target):
            t = t.to(self.device)
            loss = F.cross_entropy(y, t)
            losses.append(loss)

        losses.append(sum(losses))
        return torch.stack(losses)


def _split_vectors(vectors, num_augments):
    """Splits the latent vectors into tensors for each subproblem.

    Splits the latent vectors according to the number of augmentations per
    image for each subproblem. It returns three tensors that contain a
    subset of the latent vectors in vecs to increase efficiency.

    Args:
        vectors      = [torch.Tensor] the latent vectors to be split
        num_augments = [torch.Tensor] number of augmentations per sub-
                                      problem with shape (BATCH_SIZE, 3)

    Returns [torch.Tensor]*3:
        The latent vectors for the grapheme_root, vowel_diacritic,
        and consonant_diacritic subproblems.
    """
    if num_augments is None:
        return vectors, vectors, vectors

    # determine the slices of the latent vectors for each subproblem
    max_augments, _ = num_augments.max(dim=1, keepdim=True)
    diffs = torch.cat((torch.zeros(1, 1).long(), max_augments))
    start_indices = torch.cumsum(diffs, dim=0)[:-1]
    slices = torch.cat((start_indices, start_indices + num_augments), dim=1)

    # determine the indices of the latent vectors for each subproblem
    graph = torch.cat([torch.arange(st, end) for st, end in slices[:, [0, 1]]])
    vowel = torch.cat([torch.arange(st, end) for st, end in slices[:, [0, 2]]])
    conso = torch.cat([torch.arange(st, end) for st, end in slices[:, [0, 3]]])

    return vectors[graph], vectors[vowel], vectors[conso]


class ZeroNet(nn.Module):

    def __init__(self, device, kernel_size=3):
        super(ZeroNet, self).__init__()

        # images are 128 * 128
        # conv channels based on practice from MNIST networks
        # input channels 1, output channels 10
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)

        # input channels 10, output channels 20,
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
        """Foward pass of the CNN.

        Args:
            x            = [torch.Tensor] images with shape (N, 1, SIZE, SIZE)
            num_augments = [torch.Tensor] number of augmentations per sub-
                                          problem with shape (BATCH_SIZE, 3)

        Returns [torch.Tensor]*3:
            Non-normalized predictions for each class for each subproblem.
        """
        # put images on GPU
        x = x.to(self.device)

        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3))
        x = x.flatten(start_dim=1)   # flatten representation
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)

        x_graph, x_vowel, x_conso = _split_vectors(x, num_augments)
        y_graph = self.fc2(x_graph)
        y_vowel = self.fc3(x_vowel)
        y_conso = self.fc4(x_conso)
        return y_graph, y_vowel, y_conso


class BengaliNet(nn.Module):
    """Model that uses ResNet-50 for intermediate layers.

    Attributes:
        device = [torch.device] device to run the model on
    """

    def __init__(self, device):
        super(BengaliNet, self).__init__()

        # convolutional layer to get required number of channels
        self.conv = nn.Conv2d(1, 3, 5, padding=2)
        self.bn = nn.BatchNorm2d(3)

        # create large pre-trained ResNet model to generate image embeddings
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # extra fully-connected layers to determine labels
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 168)
        self.fc3 = nn.Linear(256, 11)
        self.fc4 = nn.Linear(256, 7)

        self.device = device
        self.to(self.device)

    def forward(self, x, num_augments=None):
        # put images on GPU
        x = x.to(self.device)

        h = self.conv(x)
        h = self.bn(h)

        h = self.resnet18(h)
        h = h.flatten(start_dim=1)

        h = self.fc1(h)
        h_graph, h_vowel, h_conso = _split_vectors(h, num_augments)
        y_graph = self.fc2(h_graph)
        y_vowel = self.fc3(h_vowel)
        y_conso = self.fc4(h_conso)
        return y_graph, y_vowel, y_conso
