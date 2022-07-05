import keras_preprocessing.image.utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import transformation as trans
import os


def display_sample(display_list, iou_score):
    """
   Show a viz of the tensor provided
   :param iou_score: iou score for true mask and prediction mask
   :param display_list: a list of tf.tensors where the order is as follow
   ['Input Image', 'True Mask', 'Predicted Mask']
   iou_score: iou score for true mask and prediction mask
   :return: None
   """

    plt.figure(figsize=(9, 4))

    title = ["Input Image", "True Mask", "Predicted Mask\n" + "iou score: " + str(f"{iou_score:.4f}")]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i], {'fontsize': 'xx-large'})
        image = display_list[i]
        image = keras_preprocessing.image.utils.array_to_img(image, scale=False)
        plt.imshow(image)
        plt.axis('off')

    plt.show()


def load_test_dataset(test_sample_paths):
    """
   Given the path of the test set, the function will load the images and mask and make them suitable for the
   deep learning model
    :param test_sample_paths:
   :return:
   """
    test_dataset = tf.data.Dataset.list_files(test_sample_paths, shuffle=False)
    test_dataset = test_dataset.map(read_image_and_find_mask)  # find respective mask
    test_dataset = test_dataset.map(resnet_test_set_processing)  # apply resnet 50 processing
    return test_dataset


def load_data_paths(training_folders):
    data_samples = np.empty(0, dtype=object)

    for folder in training_folders:
        path_of_samples = [
            os.path.abspath(os.path.join(folder, "images", file)) for file in os.listdir(os.path.join(folder, "images"))
        ]
        data_samples = np.append(data_samples, path_of_samples)

    filtered_data_samples = [samples for samples in data_samples if samples.endswith(".png")]
    # data_samples = np.sort(data_samples, kind='mergesort')
    data_samples = np.sort(filtered_data_samples, kind='mergesort')
    return data_samples


def load_training_validation_dataset(training, validation, seed):

    # take the training set and apply transformation, resizing, and preprocessing
    train_dataset = tf.data.Dataset.list_files(training, shuffle=True, seed=seed)
    train_dataset = train_dataset.map(read_image_and_find_mask)
    train_dataset = train_dataset.map(resnet_training_set_processing)

    # take the val and apply resizing and preprocessing for res net 50 backbone
    val_dataset = tf.data.Dataset.list_files(validation, shuffle=False)
    val_dataset = val_dataset.map(read_image_and_find_mask)
    val_dataset = val_dataset.map(resnet_test_set_processing)

    dataset = {"train": train_dataset, "val": val_dataset}
    return dataset


@tf.function
def resnet_test_set_processing(datapoint: dict) -> tuple:
    # makes the images compatible with the model
    input_image = tf.image.resize(datapoint['image'], (256, 256))
    input_image = tf.keras.applications.resnet50.preprocess_input(input_image)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (256, 256))
    return input_image, input_mask


@tf.function
def resnet_training_set_processing(datapoint: dict) -> tuple:
    # resize image
    resized_image = tf.image.resize(datapoint['image'], (256, 256))
    resized_mask = tf.image.resize(datapoint['segmentation_mask'], (256, 256))

    # after resizing, i will be able to apply random augmentation to the image
    transformed_image, transformed_mask = trans.rand_flip_image_vertically(resized_image, resized_mask)
    transformed_image, transformed_mask = trans.rand_flip_image_horizontally(transformed_image, transformed_mask)
    transformed_image, transformed_mask = trans.rand_rotate_image(transformed_image, transformed_mask)

    # we will next preprocess the input
    preprocessed_image = tf.keras.applications.resnet50.preprocess_input(transformed_image)
    return preprocessed_image, transformed_mask


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