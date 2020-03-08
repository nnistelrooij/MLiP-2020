from math import ceil
import random

import torch


class DropInfo(object):
    """Class to inherent information dropping algorithms from.

    Attributes:
        prob = [float] probability of using information dropping algorithm
    """

    def __init__(self):
        self.prob = 0

    def __call__(self, image):
        return image


class Cutout(DropInfo):
    """Class to augment images with cutout: https://arxiv.org/abs/1708.04552.

    Attributes:
        num_squares = [int] number of squares to cut out of the image
        length      = [int] the length (in pixels) of each square
        prob        = [float] probability of using cutout
    """

    def __init__(self, num_squares, length):
        """Initialize cutout augmentation.

        Args:
            num_squares = [int] number of squares to cut out of the image
            length      = [int] the length (in pixels) of each square
        """
        super(Cutout, self).__init__()

        self.num_squares = num_squares
        self.length = length

    def __call__(self, image):
        """Randomly mask out one or more squares from an image.

        Args:
            image = [torch.Tensor] image of shape (1, size, size)

        Returns [torch.Tensor]:
            Image with num_squares of dimension length x length cut out of it.
        """
        if self.prob < random.random():
            return image

        # determine center of squares
        image_size = image.size(-1)
        coords = torch.randint(image_size, size=(2, self.num_squares))

        # determine top-left and bottom-right corners of squares
        x1, y1 = torch.clamp(coords - self.length // 2, 0, image_size)
        x2, y2 = torch.clamp(coords + self.length // 2, 0, image_size)

        # cut squares out of image
        for x1, y1, x2, y2 in zip(x1, y1, x2, y2):
            image[:, y1:y2, x1:x2] = 0

        return image


class GridMask(DropInfo):
    """Class to augment images with GridMask: https://arxiv.org/abs/2001.04086.

    Attributes:
        keep_ratio = [int] ratio of input image pixels to preserve
        d_min      = [int] minimum length (in pixels) of one unit in grid
        d_max      = [int] maximum length (in pixels) of one unit in grid
        prob       = [float] probability of using GridMask
    """

    def __init__(self, keep_ratio, d_min, d_max):
        """Initialize GridMask augmentation.

        Args:
            keep_ratio = [int] ratio of input image pixels to preserve
            d_min      = [int] minimum length (in pixels) of one unit in grid
            d_max      = [int] maximum length (in pixels) of one unit in grid
        """
        super(GridMask, self).__init__()

        self.keep_ratio = keep_ratio
        self.d_min = d_min
        self.d_max = d_max

    def __call__(self, image):
        """Randomly mask out a structured grid of squares from an image.

        A random unit length (d) is taken between d_min and d_max (inclusive).
        Then random origin x and y coordinates for the units are taken between
        0 and d (exclusive). Given the origin, a black square is cut out of
        each unit in the bottom-right corner where the ratio of pixels preserved
        in each unit is equal to keep_ratio. This augmentation only occurs with
        probability self.prob. Otherwise, it returns the unchanged image.

        Args:
            image = [torch.Tensor] image of shape (1, size, size)

        Returns [torch.Tensor]:
            Image with structured grid of squares cut out with probability
            self.prob. Otherwise, the unchanged image is returned.
        """
        if self.prob < random.random():
            return image

        d = random.randint(self.d_min, self.d_max)  # length of unit in pixels
        delta_x = random.randrange(d)  # top-left x coordinate of origin unit
        delta_y = random.randrange(d)  # top-left y coordinate of origin unit
        l = ceil(d * self.keep_ratio)  # length of edge in unit to black square

        # copy settings of image to mask
        image_size = image.size(-1)
        mask = torch.zeros_like(image)

        # remove columns from mask
        for x in range(delta_x - d, image_size, d):
            x_start = max(x, 0)
            x_end = max(x + l, 0)
            mask[:, :, x_start:x_end] = 1

        # remove rows from mask
        for y in range(delta_y - d, image_size, d):
            y_start = max(y, 0)
            y_end = max(y + l, 0)
            mask[:, y_start:y_end] = 1

        # squares that are left over will be removed from image
        return image * mask
