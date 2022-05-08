import os
import keras_preprocessing
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
import dataset_functions
import segmentation_models as sm
import iou_score_metric
import keras_preprocessing.image.utils
import constant_values


def entire_region_evaluate(model_name, test_folders_neg, test_folders_pos, batch_size):
    UNET_model = keras.models.load_model(
        model_name, custom_objects={
            'my_iou_metric': iou_score_metric.my_iou_metric,
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


def model_evaluate(model_name, test_folders, batch_size, save, save_parameter):
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

    UNET_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss=sm.losses.dice_loss,
                       metrics=[iou_score_metric.my_iou_metric])

    results = UNET_model.evaluate(x=test_image_data, batch_size=batch_size)  # makes prediction on whole test set
    print("loss: " + str(results[0]))  # printing the loss and iou_score
    print("iou_score: " + str(results[1]))
    print("Dice_score: " + str(results[3]))


def model_inference(model_name, image_chips_folder, save_path):
    data_samples = os.listdir(os.path.join(image_chips_folder[0], "images"))
    filtered_data_samples = [samples for samples in data_samples if samples.endswith(".png")]
    filtered_data_samples = sorted(filtered_data_samples)

    UNET_model = keras.models.load_model(
        model_name,
        custom_objects={
            'my_iou_metric': iou_score_metric.my_iou_metric,
            'dice_loss': sm.losses.dice_loss
        }
    )

    # finds all images in the test set
    test_images_label = dataset_functions.load_data_paths(image_chips_folder)
    test_images = dataset_functions.load_test_dataset(test_images_label)
    test_images = test_images.batch(32)

    count = 0
    for batch_images, batch_mask in test_images:  # for all the images in test set

        batch_predictions = UNET_model.predict(batch_images)  # make a prediction

        for i in batch_predictions:  # plot prediction with the input image and ground truth\
            pixel_wise_predictions = i.round()
            image = keras_preprocessing.image.utils.array_to_img(pixel_wise_predictions, scale=False)
            name = filtered_data_samples[count]
            image_name = save_path + str(name)
            plt.rcParams["figure.figsize"] = (1, 1)
            plt.rcParams["figure.dpi"] = 256
            plt.imsave(image_name, image)
            plt.close()
            count += 1

"""
entire_region_evaluate(model_name="/home/jchavez/model/dozerline_extractor/unet/lofo/experiment1/fold_1_model",
                       test_folders_neg=[constant_values.north_nunns_imper_lidar["negative"]],
                       test_folders_pos=[constant_values.north_nunns_imper_lidar["positive"]],
                       batch_size=32)

entire_region_evaluate(model_name="/home/jchavez/model/dozerline_extractor/unet/lofo/experiment1/fold_2_model",
                       test_folders_neg=[constant_values.north_tubbs_imper_lidar["negative"]],
                       test_folders_pos=[constant_values.north_tubbs_imper_lidar["positive"]],
                       batch_size=32)

entire_region_evaluate(model_name="/home/jchavez/model/dozerline_extractor/unet/lofo/experiment1/fold_3_model",
                       test_folders_neg=[constant_values.south_nunns_imper_lidar["negative"]],
                       test_folders_pos=[constant_values.south_nunns_imper_lidar["positive"]],
                       batch_size=32)

entire_region_evaluate(model_name="/home/jchavez/model/dozerline_extractor/unet/lofo/experiment1/fold_4_model",
                       test_folders_neg=[constant_values.south_tubbs_imper_lidar["negative"]],
                       test_folders_pos=[constant_values.south_tubbs_imper_lidar["positive"]],
                       batch_size=32)

entire_region_evaluate(model_name="/home/jchavez/model/dozerline_extractor/unet/lofo/experiment1/fold_5_model",
                       test_folders_neg=[constant_values.pocket_imper_lidar["negative"]],
                       test_folders_pos=[constant_values.pocket_imper_lidar["positive"]],
                       batch_size=32)
"""

model_inference("/home/jchavez/model/dozerline_extractor/unet/lofo/experiment1/fold_1_model",
                [constant_values.north_nunns_imper_lidar["negative"]],
                "/home/jchavez/prediction/lidar_folds/fold1/")

model_inference("/home/jchavez/model/dozerline_extractor/unet/lofo/experiment1/fold_2_model",
                [constant_values.north_tubbs_imper_lidar["negative"]],
                "/home/jchavez/prediction/lidar_folds/fold2/")

model_inference("/home/jchavez/model/dozerline_extractor/unet/lofo/experiment1/fold_3_model",
                [constant_values.south_nunns_imper_lidar["negative"]],
                "/home/jchavez/prediction/lidar_folds/fold3/")

model_inference("/home/jchavez/model/dozerline_extractor/unet/lofo/experiment1/fold_4_model",
                [constant_values.south_tubbs_imper_lidar["negative"]],
                "/home/jchavez/prediction/lidar_folds/fold4/")

model_inference("/home/jchavez/model/dozerline_extractor/unet/lofo/experiment1/fold_5_model",
                [constant_values.pocket_imper_lidar["negative"]],
                "/home/jchavez/prediction/lidar_folds/fold5/")

