import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize(df):
    """Normalize the given images.

    Args:
        df = [DataFrame] images as a Pandas DataFrame

    Returns [ndarray]:
        Images normalized in a (N, HEIGHT, WIDTH) Numpy array.
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

    return img_array.reshape((len(df), HEIGHT, WIDTH)).astype(np.uint8)


def bounding_boxes(images):
    """Returns the bounding boxes around the relevant pixels.

    Args:
        images = [ndarray] the images as a (N, HEIGHT, WIDTH) Numpy array

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
    xmax = WIDTH - np.argmax(cols[:, ::-1], axis=1)
    ymin = np.argmax(rows, axis=1)
    ymax = HEIGHT - np.argmax(rows[:, ::-1], axis=1)

    # widen the bounding boxes if they are cropped too much
    xmin = (xmin - 13) * (xmin > 13)
    xmax = (xmax + 13 - WIDTH) * (xmax < WIDTH - 13) + WIDTH

    # lengthen the bounding boxes if they are cropped too much
    ymin = (ymin - 10) * (ymin > 10)
    ymax = (ymax + 10 - HEIGHT) * (ymax < HEIGHT - 10) + HEIGHT

    return np.stack((xmin, xmax, ymin, ymax), axis=1)


def crop_pad_resize(images, bboxes, size=SIZE, pad=16):
    """Crops, pads, and resizes the given images.

    Args:
        images = [ndarray] the images as (N, HEIGHT, WIDTH) Numpy array
        bboxes = [ndarray] the bounding boxes as a (N, 4) Numpy array
        size   = [int] the size of the output images
        pad    = [int] number of pixels to pad the bounding boxes of the images

    Returns [ndarray]:
        Input images cropped, padded, and resized as (N, size, size) ndarray.
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
        img_crop_pad_resize = cv2.resize(img_crop_pad, (size, size))
        images_cropped_padded_resized.append(img_crop_pad_resize)

    return np.stack(images_cropped_padded_resized)


HEIGHT = 137
WIDTH = 236
SIZE = 128
BATCH_SIZE = 512

TRAIN = ['kaggle/input/bengaliai-cv19/train_image_data_0.parquet',
         'kaggle/input/bengaliai-cv19/train_image_data_1.parquet',
         'kaggle/input/bengaliai-cv19/train_image_data_2.parquet',
         'kaggle/input/bengaliai-cv19/train_image_data_3.parquet']
OUT_TRAIN = f'train_image_data_{SIZE}.npy'

if __name__ == '__main__':
    preprocessed_train_images = []
    for file_name in TRAIN:
        # read images from parquet file
        df = pd.read_parquet(file_name)

        for batch_idx in tqdm(range(0, len(df), BATCH_SIZE)):
            # select batch of images to process
            batch = df.iloc[batch_idx:batch_idx + BATCH_SIZE]

            # process images
            normalized_images = normalize(batch)
            bboxes = bounding_boxes(normalized_images)
            preprocessed_images = crop_pad_resize(normalized_images, bboxes)

            preprocessed_train_images.append(preprocessed_images)

    # put all processed images in one big ndarray
    preprocessed_train_images = np.concatenate(preprocessed_train_images)

    # determine mean and standard deviation for normalization purposes
    mean = preprocessed_train_images.mean()
    std = preprocessed_train_images.std()
    print(f'Mean: {mean}\tStandard Deviation: {std}')

    # save images ndarray on storage for easy re-use
    np.save(OUT_TRAIN, preprocessed_train_images)
