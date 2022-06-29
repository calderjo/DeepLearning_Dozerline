import tensorflow as tf
from tensorflow import keras
import dataset_functions
import segmentation_models as sm
import iou_score_metric
import os
from osgeo import gdal
import numpy as np


def entire_region_evaluate(model_name, test_folders_neg, test_folders_pos, batch_size):

    UNET_model = keras.models.load_model(
        model_name,
        custom_objects={
            'my_iou_metric': iou_score_metric.my_iou_metric,
            'binary_crossentropy_plus_dice_loss': sm.losses.bce_dice_loss
        }
    )

    UNET_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss=sm.losses.bce_dice_loss,
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
    os.environ['PROJ_LIB'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/proj/'
    os.environ['GDAL_DATA'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/'

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

        if count == 1:
            break

        batch_predictions = UNET_model.predict(batch_images)  # make a prediction

        for prediction in batch_predictions:  # plot prediction with the input image and ground truth

            name = filtered_data_samples[count]
            image_name = save_path + str(name)
            gPNG = gdal.Open(os.path.join(image_chips_folder[0], "images", str(name)))

            size = len(image_name)  # text length
            replacement = "tif"  # replace with this
            image_name = image_name.replace(image_name[size - 3:], replacement)

            prediction = np.reshape(prediction, (256, 256))

            output_raster = gdal.GetDriverByName('GTiff').Create(image_name, 256, 256, 1,
                                                                 gdal.GDT_Float32)  # Open the file
            output_raster.SetGeoTransform(gPNG.GetGeoTransform())  # Specify its coordinates
            output_raster.SetProjection(gPNG.GetProjection())  # Exports the coordinate system

            output_raster.GetRasterBand(1).WriteArray(prediction)  # Writes my array to the raster

            output_raster.FlushCache()
            count += 1