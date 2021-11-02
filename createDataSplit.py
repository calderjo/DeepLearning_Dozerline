import matplotlib.pyplot as plt
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import glob

"""
Cool thing is that tensor flow dataset have a map function where an operation passed can be performed
across the entire dataset
"""


def load_image_dataset(path_dataset, seed):
    """
    :param
    path_dataset: takes in the path of the image dataset assume that directory is in the form
                dataset:
                        mages:
                        labels:
    seed: select seed for shuffle
    :return: return tensor flow dataset
    """

    train_dataset = tf.data.Dataset.list_files(path_dataset + "train/images/*.png", seed=seed)
    train_dataset = train_dataset.map(parse_image)
    train_dataset = train_dataset.map(load_image_train)

    # after mapping parse_image, dataset now holds both image and mask
    val_dataset = tf.data.Dataset.list_files(path_dataset + "val/images/*.png", shuffle=True, seed=seed)
    val_dataset.map(parse_image)
    val_dataset.map(load_image_test)

    dataset = {"train": train_dataset, "val": val_dataset}

    return dataset


def parse_image(img_path: str) -> dict:
    """
    :param img_path : str - Image (not the mask) location.
    :return dict: Dictionary mapping an image and its annotation.
    Load an image and its annotation (mask) and returning
    a dictionary.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "images", "labels")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)

    return {'image': image, 'segmentation_mask': mask}


@tf.function
def load_image_train(datapoint: dict) -> tuple:
    # here we normalize the training, future we could add reproducible transformation
    input_image = tf.image.resize(datapoint['image'], (572, 572))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (572, 572))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


@tf.function
def load_image_test(datapoint: dict) -> tuple:
    # Here we normalize the test/validation set, future will have more preprocessing
    input_image, input_mask = normalize(datapoint['image'], datapoint['segmentation_mask'])
    return input_image, input_mask


@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    # This is the method that the article uses to normalize the dataset
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        image = display_list[i]
        image = tf.keras.preprocessing.image.array_to_img(image)
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
