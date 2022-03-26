import os

import keras_preprocessing
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
import dataset_functions
import segmentation_models as sm
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
import iou_score_metric
from PIL import Image
import pathlib
import keras_preprocessing.image.utils

"""
def evaluate_model(model_name, test_folders_neg, test_folders_pos, batch_size):

    UNET_model = keras.models.load_model(
        model_name, custom_objects={
            'my_iou_metric': sm.metrics.IOUScore,
            'dice_loss': sm.losses.dice_loss
        }
    )

    UNET_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss=sm.losses.dice_loss,
                       metrics=[iou_score_metric.my_iou_metric])

    test_sample_paths_com = dataset_functions.load_data_paths([test_folders_neg[0], test_folders_pos[0]])
    test_com_image_data = dataset_functions.load_test_dataset(
        test_sample_paths_com)  # apply pre-processing for resnet 50

    test_com_image_data = test_com_image_data.batch(batch_size)  # same as training
    results_com = UNET_model.evaluate(x=test_com_image_data, batch_size=batch_size)

    test_sample_paths_neg = dataset_functions.load_data_paths(test_folders_neg)
    test_neg_image_data = dataset_functions.load_test_dataset(
        test_sample_paths_neg)  # apply pre-processing for resnet 50

    test_neg_image_data = test_neg_image_data.batch(batch_size)  # same as training
    results_neg = UNET_model.evaluate(x=test_neg_image_data, batch_size=batch_size)

    test_sample_paths_pos = dataset_functions.load_data_paths(test_folders_pos)
    test_pos_image_data = dataset_functions.load_test_dataset(
        test_sample_paths_pos)  # apply pre-processing for resnet 50

    test_pos_image_data = test_pos_image_data.batch(batch_size)  # same as training
    results_pos = UNET_model.evaluate(x=test_pos_image_data, batch_size=batch_size)

    print("testing: " + str(model_name) + "\n\n")

    print("Non Dirt Road Sample")
    print("neg_loss: " + str(results_neg[0]))  # printing the loss and iou_score
    print("neg_IOU_score: " + str(results_neg[1]))
    test_neg_image_data = None

    print("Dirt Road Sample")
    print("poss_loss: " + str(results_pos[0]))  # printing the loss and iou_score
    print("poss_IOU_score: " + str(results_pos[1]))
    test_pos_image_data = None

    print("Combined Samples")
    print("com_loss: " + str(results_com[0]))  # printing the loss and iou_score
    print("com_IOU_score: " + str(results_com[1]))
    test_com_image_data = None

    print("\n\n Test Finished")
"""


def evaluate_model(model_name, test_folders, batch_size, visualize):

    test_sample_paths = dataset_functions.load_data_paths(test_folders)
    test_image_data = dataset_functions.load_test_dataset(test_sample_paths)  # apply pre-processing for resnet 50
    test_image_data = test_image_data.batch(batch_size)  # same as training

    UNET_model = keras.models.load_model(
        model_name,
        custom_objects={
            'iou_score': sm.metrics.IOUScore,
            'dice_loss': sm.losses.dice_loss
        }
    )

    """
        UNET_model = keras.models.load_model(
        model_name,
        custom_objects={
            'iou_score': sm.metrics.IOUScore,
            'my_iou_metric': sm.metrics.IOUScore,
            'dice_loss': sm.losses.dice_loss
        }
    )
    
    
        UNET_model = keras.models.load_model(
        model_name,
        custom_objects={
            'iou_score': sm.metrics.IOUScore,
            'dice_loss': sm.losses.dice_loss
        }
    )
    """

    UNET_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss=sm.losses.dice_loss,
                       metrics=[iou_score_metric.my_iou_metric])

    save_all_prediction("/home/jchavez/dataset/Dirt_Gravel_Roads/North_Tubbs/positive_samples/images",
                        test_image_data,
                        UNET_model,
                        "/home/jchavez/prediction/imper_data/positive/")

    # results = UNET_model.evaluate(x=test_image_data, batch_size=batch_size)  # makes prediction on whole test set

    # print("loss: " + str(results[0]))  # printing the loss and iou_score
    # print("fast_iou_score: " + str(results[1]))
    # print("iou_score: " + str(results[2]))
    # print("Dice_score: " + str(results[3]))

    # if visualize
    #    visualize_predictions(test_image_data, UNET_model, test_folders[0])


def save_all_prediction(test_set_path, test_image_data, UNET_model, save_path):
    # finds all images in the test set
    fileExt = r"*.png"
    test_images = os.listdir(test_set_path)
    test_images_sorted = sorted(test_images)

    count = 0
    for batch_images, batch_mask in test_image_data:  # for all the images in test set
        batch_predictions = UNET_model.predict(batch_images)  # make a prediction
        for i in batch_predictions:  # plot prediction with the input image and ground truth\
            pixel_wise_predictions = i.round()
            image = keras_preprocessing.image.utils.array_to_img(pixel_wise_predictions, scale=False)
            name = test_images_sorted[count]
            image_name = save_path + str(name)
            plt.rcParams["figure.figsize"] = (1, 1)
            plt.rcParams["figure.dpi"] = 256
            plt.imsave(image_name, image)
            plt.close()
            count += 1


def visualize_predictions(test_image_data, UNET_model, test_set_path):
    # finds all images in the test set

    fileExt = r"*.png"
    files = list(pathlib.Path(test_set_path).glob(fileExt))
    test_images = sorted(files)
    count = 0
    for image, mask in test_image_data.take(64):  # for all the images in test set
        predictions = UNET_model.predict(image)  # make a prediction
        for i in range(0, 32):  # plot prediction with the input image and ground truth\

            m = np.reshape(mask[i].numpy(), (1, 256, 256, 1), order='C')
            p = np.reshape(predictions[i], (1, 256, 256, 1), order='C')
            prediction_score = iou_score_metric.my_iou_metric(m, p)
            predictions = UNET_model.predict(image).round()
            input_image = tf.io.read_file((os.path.join(test_set_path, "images", test_images[count])))
            input_image = tf.image.decode_png(input_image, channels=3, dtype=tf.uint16)
            input_image = tf.image.convert_image_dtype(input_image, tf.uint8)

            dataset_functions.display_sample([input_image, mask[i], predictions[i]], prediction_score)
            count += 1


# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_landcover/resnet18/experiment_3/trial_2/trial_2_model",
    test_folders=["/home/jchavez/dataset/Dirt_Gravel_Roads/North_Tubbs/positive_samples/"],
    batch_size=32,
    visualize=True
)



"""
evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_0_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_1_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_2_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_3_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_4_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_5_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_6_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_7_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_8_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_9_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_0_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_1_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_2_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_3_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_4_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_5_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_6_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_7_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_8_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_9_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_0_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_1_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_2_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_3_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_4_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_5_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_6_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_7_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_8_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_9_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_0_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_1_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_2_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_3_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_4_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_5_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_6_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_7_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_8_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_9_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_0_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_1_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_2_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_3_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_4_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_5_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_6_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_7_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_8_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_9_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_0_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_1_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_2_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_3_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_4_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_5_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_6_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_7_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_8_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_9_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_0_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_1_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_2_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_3_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_4_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_5_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_6_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_7_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_8_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_9_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_0_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_1_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_2_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_3_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_4_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_5_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_6_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_7_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_8_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_9_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_0_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_1_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_2_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_3_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_4_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_5_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_6_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_7_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_8_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_9_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_0_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_1_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_2_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_3_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_4_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_5_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_6_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_7_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_8_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_9_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_0_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_1_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_2_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_3_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_4_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_5_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_6_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_7_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_8_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_9_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_0_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_1_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_2_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_3_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_4_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_5_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_6_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_7_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_8_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_9_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

"""
