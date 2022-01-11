import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import dataset_functions
import segmentation_models as sm
from sklearn.metrics import jaccard_score
import argparse


def evaluate_model(model_name, test_folders, batch_size, visualize):

    test_sample_paths = dataset_functions.load_data_paths(test_folders)
    test_image_data = dataset_functions.load_test_dataset(test_sample_paths)  # apply pre-processing for resnet 50
    test_image_data = test_image_data.batch(batch_size)  # same as training

    UNET_model = keras.models.load_model(
        model_name,
        custom_objects={
            'iou_score': sm.metrics.iou_score,
            'dice_loss': sm.losses.dice_loss
        }
    )

    results = UNET_model.evaluate(x=test_image_data, batch_size=batch_size)  # makes prediction on whole test set

    print("loss: " + str(results[0]))  # printing the loss and iou_score
    print("iou_score: " + str(results[1]))

    if visualize:
        visualize_predictions(test_image_data, UNET_model, test_folders[0])


def visualize_predictions(test_image_data, UNET_model, test_set_path):
    # finds all images in the test set
    files = os.listdir(os.path.join(test_set_path, "images"))
    test_images = sorted(files)

    count = 0
    for image, mask in test_image_data.take(4):  # for all the images in test set
        predictions = UNET_model.predict(image).round()  # make a prediction
        for i in range(0, 32):  # plot prediction with the input image and ground truth

            mask_val = np.ravel(mask[i].numpy(), order='C')
            pred_val = np.ravel(predictions[i], order='C')

            prediction_score = jaccard_score(mask_val, pred_val)

            input_image = tf.io.read_file((os.path.join(test_set_path, "images", test_images[count])))
            input_image = tf.image.decode_png(input_image, channels=3, dtype=tf.uint16)
            input_image = tf.image.convert_image_dtype(input_image, tf.uint8)

            dataset_functions.display_sample([input_image, mask[i], predictions[i]], prediction_score)
            count += 1


evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/resnet101/experiment_2/trial_3_model",
    test_folders=["/home/jchavez/dataset/Dirt_Gravel_Roads_False_Image/North_Tubbs/positive_samples"],
    batch_size=32,
    visualize=True
)
