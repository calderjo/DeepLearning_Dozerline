import os
import numpy as np
import tensorflow as tf
from tensorflow import data

"""
Cool thing is that tensor flow dataset have a map function where an operation passed can be performed
across the entire dataset
"""


def parse_image(img_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.
    :param
    img_path : str - Image (not the mask) location.

    :returns
    dict: Dictionary mapping an image and its annotation.
    """

    image = tf.io.read_file(img_path)  # reads in an image
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "images", "labels")  # these two lines find the associated mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)  # remaps 255 - is dozer to 1

    return {'image': image, 'segmentation_mask': mask}


def load_image_dataset(path_dataset, seed):
    """
    :param
    path_dataset: takes in the path of the image dataset assume that directory is in the form
        dataset:
            images:
            labels:
    seed: select seed for shuffle
    :return: return tensor flow dataset
    """

    dataset = tf.data.Dataset.list_files(path_dataset + "images/*.png", shuffle=True, seed=seed)
    dataset.map(parse_image)  # after mapping parse_image, dataset now holds both image and mask

    dataset.shuffle(buffer_size=dataset)