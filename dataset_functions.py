import keras_preprocessing.image.utils
import matplotlib.pyplot as plt
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import glob
import segmentation_models as sm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as data_aug
import random


def load_training_validation_dataset(path_dataset, seed):

    train_dataset = tf.data.Dataset.list_files(path_dataset + "train/images/*.png", seed=seed)
    train_dataset = train_dataset.map(read_image_and_find_mask)
    train_dataset = train_dataset.map(training_set_processing)

    val_dataset = tf.data.Dataset.list_files(path_dataset + "val/images/*.png", shuffle=True, seed=seed)
    val_dataset = val_dataset.map(read_image_and_find_mask)
    val_dataset = val_dataset.map(test_set_processing)

    dataset = {"train": train_dataset, "val": val_dataset}

    return dataset


def read_image_and_find_mask(img_path: str) -> dict:
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
    test_dataset = tf.data.Dataset.list_files(path_dataset + "/images/*.png", shuffle=True, seed=seed)
    test_dataset = test_dataset.map(read_image_and_find_mask)
    test_dataset = test_dataset.map(test_set_processing)

    return test_dataset


@tf.function
def training_set_processing(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (512, 512))
    input_image = tf.keras.applications.resnet50.preprocess_input(input_image)

    input_mask = tf.image.resize(datapoint['segmentation_mask'], (512, 512))

    # after resizing, i will be able to apply augmentation
    return input_image, input_mask


@tf.function
def test_set_processing(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (512, 512))
    input_image = tf.keras.applications.resnet50.preprocess_input(input_image)

    input_mask = tf.image.resize(datapoint['segmentation_mask'], (512, 512))
    return input_image, input_mask


@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    # This is the method that the article uses to normalize the dataset
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


def display_sample(display_list):

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
