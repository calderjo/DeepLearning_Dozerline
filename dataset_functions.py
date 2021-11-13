import keras_preprocessing.image.utils
import matplotlib.pyplot as plt
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import glob


def load_training_validation_dataset(path_dataset, seed):

    # take the training set and apply transformation, resizing, and preprocessing
    train_dataset = tf.data.Dataset.list_files(path_dataset + "train/images/*.png", seed=seed)
    train_dataset = train_dataset.map(read_image_and_find_mask)
    train_dataset = train_dataset.map(training_set_processing)

    # take the val and apply resizing and preprocessing for res net 50 backbone
    val_dataset = tf.data.Dataset.list_files(path_dataset + "val/images/*.png", shuffle=True, seed=seed)
    val_dataset = val_dataset.map(read_image_and_find_mask)
    val_dataset = val_dataset.map(test_set_processing)

    dataset = {"train": train_dataset, "val": val_dataset}

    return dataset


def read_image_and_find_mask(img_path: str) -> dict:

    """
    Given the path of the image directory, the function will load the image and will convert it to a
    uint8 image and find it respective mask and convert that to an 8 bit uint8 image
    :param img_path: directory path to image directory
    :return: returns dictionary of the image and mask together
    """

    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3, dtype=tf.uint16)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "images", "labels")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1, dtype=tf.uint16)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)

    mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)
    return {'image': image, 'segmentation_mask': mask}


def load_test_dataset(path_dataset, seed):
    """
    Given the path of the test set, the function will load the images and mask and make them suitable for the
    deep learning model

    :param path_dataset:
    :param seed:
    :return:
    """
    test_dataset = tf.data.Dataset.list_files(path_dataset + "/images/*.png", shuffle=True, seed=seed)
    test_dataset = test_dataset.map(read_image_and_find_mask)
    test_dataset = test_dataset.map(test_set_processing)
    return test_dataset


@tf.function
def training_set_processing(datapoint: dict) -> tuple:

    # resize image
    resized_image = tf.image.resize(datapoint['image'], (512, 512))
    resized_mask = tf.image.resize(datapoint['segmentation_mask'], (512, 512))

    # after resizing, i will be able to apply random augmentation to the image
    transformed_image, transformed_mask = rand_flip_image_vertically(resized_image, resized_mask)
    transformed_image, transformed_mask = rand_flip_image_horizontally(transformed_image, transformed_mask)

    # we will next preprocess the input
    preprocessed_image = tf.keras.applications.resnet50.preprocess_input(transformed_image)

    return preprocessed_image, transformed_mask


@tf.function
def test_set_processing(datapoint: dict) -> tuple:
    # makes the images compatible with the model
    input_image = tf.image.resize(datapoint['image'], (512, 512))
    input_image = tf.keras.applications.resnet50.preprocess_input(input_image)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (512, 512))
    return input_image, input_mask


@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    # This is the method that the article uses to normalize the dataset
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


@tf.function
def rand_flip_image_vertically(input_image: tf.Tensor, input_mask: tf.Tensor):
    """Randomly flip an image and label vertically (upside down)."""

    uniform_random = tf.random.uniform([], 0, 1.0)  # select a rand number

    flip_cond = tf.less(uniform_random, .5)  # if number is less than .5, then we will preform a flip
    image = tf.cond(flip_cond, lambda: tf.image.flip_up_down(input_image), lambda: input_image)
    mask = tf.cond(flip_cond, lambda: tf.image.flip_up_down(input_mask), lambda: input_mask)

    return image, mask


@tf.function
def rand_flip_image_horizontally(input_image: tf.Tensor, input_mask: tf.Tensor):
    """Randomly flip an image and label horizontally"""

    uniform_random = tf.random.uniform([], 0, 1.0)  # select a rand number

    flip_cond = tf.less(uniform_random, .5)  # if number is less than .5, then we will preform a flip
    image = tf.cond(flip_cond, lambda: tf.image.flip_left_right(input_image), lambda: input_image)
    mask = tf.cond(flip_cond, lambda: tf.image.flip_left_right(input_mask), lambda: input_mask)

    return image, mask


def display_sample(display_list):
    """
    Show a viz of the tensor provided
    :param display_list: a list of tf.tensors where the order is as follow
    ['Input Image', 'True Mask', 'Predicted Mask']
    :return: None
    """

    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        image = display_list[i]
        image = keras_preprocessing.image.utils.array_to_img(image, scale=False)
        plt.imshow(image)
        plt.axis('off')

    plt.show()


def create_training_validation(source_path, destination_path, validation_ratio, rand_seed):
    # find all images and label in our folder created by arcgis pro
    source_path_image_chip = sorted(list(glob.glob(os.path.join(source_path, "images/*.png"))))
    source_path_label = sorted(list(glob.glob(os.path.join(source_path, "labels/*.png"))))

    train_data, val_data, train_label, val_label = train_test_split(source_path_image_chip,
                                                                    source_path_label,
                                                                    test_size=validation_ratio,
                                                                    random_state=rand_seed,
                                                                    shuffle=True)

    # create  directory with train and validation for image and label chips
    sub_folder_names = ["train", "val"]
    for sub_folder_name in sub_folder_names:
        if not os.path.exists(os.path.join(destination_path, sub_folder_name, "images")):
            os.makedirs(os.path.join(destination_path, sub_folder_name, "images"))
        else:
            shutil.rmtree(os.path.join(destination_path, sub_folder_name, "images"))
            os.makedirs(os.path.join(destination_path, sub_folder_name, "images"))
        if not os.path.exists(os.path.join(destination_path, sub_folder_name, "labels")):
            os.makedirs(os.path.join(destination_path, sub_folder_name, "labels"))
        else:
            shutil.rmtree(os.path.join(destination_path, sub_folder_name, "labels"))
            os.makedirs(os.path.join(destination_path, sub_folder_name, "labels"))

    # make a copy of each image in their area
    for file in train_data:
        shutil.copy(file, os.path.join(destination_path, "train/images"))
    for file in train_label:
        shutil.copy(file, os.path.join(destination_path, "train/labels"))
    for file in val_data:
        shutil.copy(file, os.path.join(destination_path, "val/images"))
    for file in val_label:
        shutil.copy(file, os.path.join(destination_path, "val/labels"))
    return
