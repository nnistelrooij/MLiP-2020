import torch


class Cutout(object):
    """Class to augment images with cutout: https://arxiv.org/abs/1708.04552.

    Attributes:
        num_squares = [int] number of squares to cut out of the image
        length      = [int] the length (in pixels) of each square
        size        = [int] the length (in pixels) of the images
    """

    def __init__(self, num_squares, length, size):
        """Initialize cutout augmentation.

        Args:
            num_squares = [int] number of squares to cut out of the image
            length      = [int] the length (in pixels) of each square
            size        = [int] the length (in pixels) of the images
        """
        self.num_squares = num_squares
        self.length = length
        self.size = size

    def __call__(self, image):
        """Randomly mask out one or more squares from an image.

        Args:
            image = [torch.Tensor] image of shape (1, size, size)

        Returns [torch.Tensor]:
            Image with num_squares of dimension length x length cut out of it.
        """
        # determine center of squares
        coords = torch.randint(high=self.size, size=(2, self.num_squares))

        # determine top-left and bottom-right corners of squares
        x1, y1 = torch.clamp(coords - self.length // 2, 0, self.size)
        x2, y2 = torch.clamp(coords + self.length // 2, 0, self.size)

        # cut squares out of image
        for x1, y1, x2, y2 in zip(x1, y1, x2, y2):
            image[:, y1:y2, x1:x2] = 0

        return image
