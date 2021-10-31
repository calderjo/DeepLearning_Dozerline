import os
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from PIL import Image, ImageOps


def file_to_numpy(directory, is_mask):

    # this list will hold numpy array of our image chips
    all_images = []
    image_names = os.listdir(directory)
    image_names = sorted(image_names)

    for imageChip in image_names:

        im = Image.open(directory + imageChip)  # load image
        gray_img = ImageOps.grayscale(im)       # grayscale image

        if is_mask:
            im_resize = im.resize((388, 388))
            arr = np.array(im_resize)

        else:  # is an image chips
            arr = np.asarray(gray_img)
            arr = arr / 255
            arr = arr.reshape([572, 572, 1])

        all_images.append(arr)

    img_data_np = np.array(all_images)
    return img_data_np


def create_data_split(image_directory, label_directory, train_ratio, val_ratio, test_ratio, seed):

    image_data = file_to_numpy(image_directory, False)
    mask_data = file_to_numpy(label_directory, True)

    # first split creates a train set with x percent of the data
    test_size = 1 - train_ratio

    train_data, test_val_data, train_label, test_val_label = \
        model_selection.train_test_split(image_data, mask_data, test_size=test_size, random_state=seed, shuffle=True)

    # first split creates a val set with y percent of data and a test set with z percent of the data
    test_size = test_ratio / (test_ratio + val_ratio)

    val_data, test_data, val_label, test_label = \
        model_selection.train_test_split(test_val_data, test_val_label, test_size=test_size, random_state=seed,
                                         shuffle=True)

    # Returning the data split
    return train_data, train_label, val_data, val_label, test_data, test_label
