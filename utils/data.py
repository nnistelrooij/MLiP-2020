import gc
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch._six import container_abcs, int_classes, string_classes
from torchvision import transforms

from utils.transforms import Cutout

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
        transform      = [Compose] applies a random affine transformation,
                                   normalizes to z-scores, and applies cutout
                                   transformation to a Numpy array image
        normalize      = [Normalize] normalizes Numpy array image to z-scores
        labels         = [torch.Tensor] images labels tensor of shape (N, 3)
        mod_counts     = [torch.Tensor] remainders of dividing each class
                                        frequency by the highest frequency
        ratio_counts   = [torch.Tensor] floors of dividing each class
                                        frequency by the highest frequency
        current_counts = [torch.Tensor] number of retrieved items of each
                                        class in current iteration of epoch
        augment        = [bool] whether or not the images are transformed
        balance        = [bool] whether or not the classes are balanced
    """

    def __init__(self, images, labels, image_size, augment=False, balance=False):
        """Initialize dataset.

        Args:
            images     = [ndarray] images array with shape (N, SIZE, SIZE)
            labels     = [DataFrame] image labels DataFrame of shape (N, 3)
            image_size = [int] the length (in pixels) of the images
            augment    = [bool] whether or not the images are transformed
            balance    = [bool] whether or not the classes are balanced
        """
        super(Dataset, self).__init__()

        # initialize transformations from torchvision.transforms
        self.images = images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=(-8, 8),
                translate=(1/24, 1/24),
                scale=(8/9, 10/9)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.071374745,), std=(0.20761949,)),
            Cutout(8, 12, image_size)
        ])
        self.normalize = transforms.Compose([
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

        self.augment = augment
        self.balance = balance

    def reset(self):
        """Reset number of retrieved items of each class in current epoch."""
        self.current_counts = torch.zeros_like(self.mod_counts)

    def __len__(self):
        return len(self.images)

    def _num_augmentations(self, labels):
        """Computes number of augmentations for given image labels.

        Args:
            labels = [torch.Tensor] image labels of shape (3,)

        Returns [torch.Tensor]:
            If self.balance is False, a tensor filled with ones is returned.
            Otherwise, the number of augmentations will ensure that all the
            classes are seen the same number of times for each subproblem.
        """
        if not self.balance:  # one augmentation
            return torch.tensor([1] * len(labels))

        # select current and modular counts for given labels
        current_counts = self.current_counts[[0, 1, 2], labels]
        self.current_counts[[0, 1, 2], labels] += 1
        mod_counts = self.mod_counts[[0, 1, 2], labels]

        # determine number of augmentations with possible extra augmentation
        extra_augment = current_counts < mod_counts
        num_augments = self.ratio_counts[[0, 1, 2], labels] + extra_augment

        return num_augments.long()

    def _augment_or_normalize(self, image):
        """Augments (including normalization) or normalizes image.

        Args:
            image = [ndarray] Numpy array image of shape (SIZE, SIZE)

        Returns [torch.Tensor]
            Augmented or normalized image with shape (1, 1, SIZE, SIZE).
        """
        if self.augment:  # random affine, normalize, cutout
            image = self.transform(image)
        else:  # normalize
            image = self.normalize(image)

        return image.unsqueeze(0)

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
        images = self._augment_or_normalize(image)
        for _ in range(max(num_augments) - 1):
            images = torch.cat((images, self._augment_or_normalize(image)))

        # repeat labels given number of augmentations
        labels = [labels[i].repeat(num_augments[i]) for i in range(len(labels))]

        # return images, labels, and number of augmentations as a 5-tuple
        return (images,) + tuple(labels) + (num_augments.unsqueeze(0),)


def load_data(images_path, labels_path, image_size, batch_size,
              split=0.2, augment=False, balance=False):
    """Load the images and labels from storage into DataLoader objects.

    Args:
        images_path = [str] path for the images .npy file
        labels_path = [str] path for the labels CSV file
        image_size  = [int] the length (in pixels) of the images
        batch_size  = [int] batch size of the DataLoader objects
        split       = [float] percentage of data used for validation
        augment     = [bool] whether or not the images are transformed
        balance     = [bool] whether or not the classes are balanced

    Returns [BengaliDataset, DataLoader, DataLoader]:
        train_dataset = data set of the training data
        train_loader  = DataLoader of the training data
        val_loader    = DataLoader of the validation data
    """
    train_images = np.load(images_path)
    train_labels = pd.read_csv(labels_path).iloc[:, 1:-1]

    # train/validation split 80/20
    train_images, val_images = train_test_split(train_images, test_size=split)
    train_labels, val_labels = train_test_split(train_labels, test_size=split)
    gc.collect()  # garbage collection

    # training set
    train_dataset = BengaliDataset(train_images, train_labels, image_size,
                                   augment=augment, balance=balance)
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=4,
                              batch_size=batch_size, collate_fn=_cat_collate)

    # validation set
    val_dataset = BengaliDataset(val_images, val_labels, image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=4,  collate_fn=_cat_collate)

    return train_dataset, train_loader, val_loader
