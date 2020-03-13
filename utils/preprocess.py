import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize(df, width, height):
    """Normalize the images in the given DataFrame.

    Args:
        df     = [DataFrame] images as a Pandas DataFrame
        width  = [int] width of the raw images in pixels
        height = [int] height of the raw images in pixels

    Returns [ndarray]:
        Images normalized in a (N, height, width) Numpy array.
    """
    # images are stored in inverse
    img_array = 255 - df.iloc[:, 1:].to_numpy()

    # make use of full grayscale spectrum
    img_min = img_array.min(axis=1, keepdims=True)
    img_array = img_array - img_min

    img_max = img_array.max(axis=1, keepdims=True)
    img_array = img_array * (255 / img_max)

    # remove low-intensity pixels
    img_array[img_array < 26] = 0

    return img_array.reshape((len(df), height, width)).astype(np.uint8)


def bounding_boxes(images, width, height):
    """Returns the bounding boxes around the relevant pixels.

    Args:
        images = [ndarray] the images as a (N, height, width) Numpy array
        width  = [int] width of the raw images in pixels
        height = [int] height of the raw images in pixels

    Returns [ndarray]:
        Left x pixel, right x pixel, top y pixel, bottom y pixel of
        the bounding box for each image.
    """
    # remove lines at the boundary of the images
    images = images[:, 5:-5, 5:-5]
    images = np.pad(images, [(0,), (5,), (5,)], mode='constant')

    # find columns and rows that have visible pixels
    cols = np.any(images > 170, axis=1)
    rows = np.any(images > 170, axis=2)

    # find first and last pixels of columns and rows, respectively
    xmin = np.argmax(cols, axis=1)
    xmax = width - np.argmax(cols[:, ::-1], axis=1)
    ymin = np.argmax(rows, axis=1)
    ymax = height - np.argmax(rows[:, ::-1], axis=1)

    # widen the bounding boxes if they are cropped too much
    xmin = (xmin - 13) * (xmin > 13)
    xmax = (xmax + 13 - width) * (xmax < width - 13) + width

    # lengthen the bounding boxes if they are cropped too much
    ymin = (ymin - 10) * (ymin > 10)
    ymax = (ymax + 10 - height) * (ymax < height - 10) + height

    return np.stack((xmin, xmax, ymin, ymax), axis=1)


def crop_pad_resize(images, bboxes, out_size, pad=16):
    """Crops, pads, and resizes the given images.

    Args:
        images   = [ndarray] the images as (N, height, width) Numpy array
        bboxes   = [ndarray] the bounding boxes as a (N, 4) Numpy array
        out_size = [int] the size of the output images in pixels
        pad      = [int] number of pixels to pad the bounding boxes

    Returns [ndarray]:
        Input images cropped, padded, and resized as
        (N, out_size, out_size) Numpy ndarray.
    """
    images_cropped_padded_resized = []
    for img, (xmin, xmax, ymin, ymax) in zip(images, bboxes):
        # crop the image
        img_crop = img[ymin:ymax, xmin:xmax]

        # compute length of square cropped image
        width = xmax - xmin
        height = ymax - ymin
        length = max(width, height) + pad

        # make sure that the aspect ratio is kept in resizing
        padding = [((length - height) // 2,), ((length - width) // 2,)]
        img_crop_pad = np.pad(img_crop, padding, mode='constant')

        # resize image to standard resolution
        img_crop_pad_resize = cv2.resize(img_crop_pad, (out_size, out_size))
        images_cropped_padded_resized.append(img_crop_pad_resize)

    return np.stack(images_cropped_padded_resized)


def preprocess(files, width, height, out_size, batch_size=512):
    """Preprocess the grapheme images in the given files.

    Args:
        files      = [list] list of file paths to the parquet files with images
        width      = [int] width of the raw images in pixels
        height     = [int] height of the raw images in pixels
        out_size   = [int] the size of the output images in pixels
        batch_size = [int] number of images to process at a time

    Returns [ndarray]:
        Preprocessed images in (N, out_size, out_size) Numpy ndarray.
    """
    preprocessed_images = []
    for file_name in files:
        # read images from parquet file
        df = pd.read_parquet(file_name)

        for batch_idx in tqdm(range(0, len(df), batch_size)):
            # select batch of images to process
            batch = df.iloc[batch_idx:batch_idx + batch_size]

            # process images
            normalized_images = normalize(batch, width, height)
            bboxes = bounding_boxes(normalized_images, width, height)
            images = crop_pad_resize(normalized_images, bboxes, out_size)

            preprocessed_images.append(images)

    # put all preprocessed images in one big ndarray
    return np.concatenate(preprocessed_images)


if __name__ == '__main__':
    # preprocess training images
    train_files = ['../kaggle/input/bengaliai-cv19/train_image_data_0.parquet',
                   '../kaggle/input/bengaliai-cv19/train_image_data_1.parquet',
                   '../kaggle/input/bengaliai-cv19/train_image_data_2.parquet',
                   '../kaggle/input/bengaliai-cv19/train_image_data_3.parquet']
    preprocessed_train_images = preprocess(train_files, 236, 137, 128)

    # determine mean and standard deviation for normalization purposes
    mean = preprocessed_train_images.mean()
    std = preprocessed_train_images.std()
    print(f'Mean: {mean}\tStandard Deviation: {std}')

    # save training images ndarray on storage for easy re-use
    np.save(f'../train_image_data_{128}.npy', preprocessed_train_images)
