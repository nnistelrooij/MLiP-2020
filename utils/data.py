import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch._six import container_abcs, int_classes, string_classes
from torchvision import transforms

from utils.transforms import DropInfo, Cutout, GridMask

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def _cat_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return _cat_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: _cat_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_cat_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [_cat_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class BengaliDataset(Dataset):
    """Class to get images and labels.

    Attributes:
        images         = [ndarray] images array with shape (N, SIZE, SIZE)
        drop_info_fn   = [object] function to drop information from images
        transform      = [Compose] transformation applied to each image
        labels         = [torch.Tensor] images labels tensor of shape (N, 3)
        mod_counts     = [torch.Tensor] remainders of dividing each class
                                        frequency by the highest frequency
        ratio_counts   = [torch.Tensor] floors of dividing each class
                                        frequency by the highest frequency
        current_counts = [torch.Tensor] number of retrieved items of each
                                        class in current iteration of epoch
        balance        = [bool] whether or not the classes are balanced
    """

    def __init__(self, images, labels, augment=False, drop_info_fn=None,
                 balance=False):
        """Initialize dataset.

        Args:
            images       = [ndarray] images array with shape (N, SIZE, SIZE)
            labels       = [DataFrame] image labels DataFrame of shape (N, 3)
            augment      = [bool] whether or not the images are transformed
            drop_info_fn = [str] whether to use cutout ('cutout'), GridMask
                                 ('gridmask'), or no info dropping algorithm
            balance      = [bool] whether or not the classes are balanced
        """
        super(Dataset, self).__init__()

        self.images = images

        # initialize information dropping algorithm
        if drop_info_fn == 'cutout':
            self.drop_info_fn = Cutout(2, 32)
        elif drop_info_fn == 'gridmask':
            self.drop_info_fn = GridMask(0.6, 28, 64)
        else:
            self.drop_info_fn = DropInfo()

        # initialize chosen transformation
        if augment:
            # initialize affine, normalizing, and info dropping transformations
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(
                    degrees=(-8, 8),
                    translate=(1/24, 1/24),
                    scale=(8/9, 10/9)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.071374745,), std=(0.20761949,)),
                self.drop_info_fn
            ])
        else:
            # initialize normalizing transformation
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.071374745,), std=(0.20761949,))
            ])

        # initialize labels and counts for class balancing
        self.labels = torch.tensor(labels.to_numpy())
        counts = labels.apply(pd.Series.value_counts).to_numpy().T
        max_counts = np.nanmax(counts, axis=1, keepdims=True)
        self.mod_counts = torch.tensor(max_counts % counts)
        self.ratio_counts = torch.tensor(max_counts // counts)
        self.current_counts = torch.zeros_like(self.mod_counts)

        self.balance = balance

    def reset(self, epoch, num_epochs):
        """Reset class balancing and information dropping algorithms.

        Args:
            epoch      = [int] current epoch of training loop starting with 1
            num_epochs = [int] number of iterations over the train data set
        """
        self.current_counts = torch.zeros_like(self.mod_counts)
        self.drop_info_fn.prob = min(epoch / num_epochs, 0.8)

    def __len__(self):
        return len(self.images)

    def _num_augmentations(self, labels, max_augments=20):
        """Computes number of augmentations for given image labels.

        Args:
            labels       = [torch.Tensor] image labels of shape (3,)
            max_augments = [int] maximum number of augmentations per image

        Returns [torch.Tensor]:
            If self.balance is False, a tensor filled with ones is returned.
            Otherwise, the number of augmentations will ensure that all the
            classes are seen the same number of times for each sub-problem
            with a maximum of max_augments augmentations per sub-problem.
        """
        if not self.balance:  # one augmentation
            return torch.tensor([1]*len(labels))

        # select current and modular counts for given labels
        current_counts = self.current_counts[[0, 1, 2], labels]
        self.current_counts[[0, 1, 2], labels] += 1
        mod_counts = self.mod_counts[[0, 1, 2], labels]

        # determine number of augmentations with possible extra augmentation
        extra_augment = current_counts < mod_counts
        num_augments = self.ratio_counts[[0, 1, 2], labels] + extra_augment

        num_augments = num_augments.clamp(max=max_augments)
        return num_augments.long()

    def __getitem__(self, idx):
        """Get images, labels, and number of augmentations.

        Args:
            idx = [int] index of original image and labels

        Returns [torch.Tensor]*5:
            images       = images tensor of shape (N, 1, SIZE, SIZE)
            labels_graph = labels tensor of grapheme_root subproblem
            labels_vowel = labels tensor of vowel_diacritic subproblem
            labels_conso = labels tensor of consonant_diacritic subproblem
            num_augments = number of augmentations of shape (1, 3)
        """
        # select image and labels
        image = self.images[idx]
        labels = self.labels[idx]

        # determine number of augmentations per subproblem
        num_augments = self._num_augmentations(labels)

        # transform or normalize image
        images = []
        for _ in range(max(num_augments)):
            images.append(self.transform(image))
        images = torch.stack(images)

        # repeat labels given number of augmentations
        labels = [label.repeat(n) for label, n in zip(labels, num_augments)]

        # return images, labels, and number of augmentations as a 5-tuple
        return (images,) + tuple(labels) + (num_augments.unsqueeze(0),)


def load_data(images_path, labels_path, test_ratio, seed, augment, drop_info_fn,
              balance, batch_size):
    """Load the images and labels from storage into DataLoader objects.

    Args:
        images_path  = [str] path for the images .npy file
        labels_path  = [str] path for the labels CSV file
        test_ratio   = [float] percentage of data used for validation
        seed         = [int] seed used for consistent data splitting
        augment      = [bool] whether or not the images are transformed
        drop_info_fn = [str] whether to use cutout ('cutout'), GridMask
                             ('gridmask'), or no info dropping algorithm
        balance      = [bool] whether or not the classes are balanced
        batch_size   = [int] batch size of the DataLoader objects

    Returns [BengaliDataset, DataLoader, DataLoader, int]:
        train_dataset = data set of the training data
        train_loader  = DataLoader of the training data
        val_loader    = DataLoader of the validation data
        image_size    = length of square images in pixels
    """
    images = np.load(images_path)
    labels = pd.read_csv(labels_path).iloc[:, 1:-1]

    # split data into train and validation splits
    splitting = train_test_split(images, labels, test_size=test_ratio,
                                 random_state=seed)
    train_images, val_images, train_labels, val_labels = splitting

    # training set
    train_dataset = BengaliDataset(train_images, train_labels,
                                   augment, drop_info_fn, balance)
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=4,
                              batch_size=batch_size, collate_fn=_cat_collate)

    # validation set
    val_dataset = BengaliDataset(val_images, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=4, collate_fn=_cat_collate)

    return train_dataset, train_loader, val_loader, images.shape[-1]
