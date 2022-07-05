import tensorflow as tf
from tensorflow import keras
import model_pre_processing
import segmentation_models as sm
import model_metrics
import numpy as np
from skimage.measure import label
from skimage import morphology
from osgeo import gdal
import os
from PIL import Image


def entire_region_evaluate(model_name, custom_objects, positive_sample, negative_sample, batch_size):

    UNET_model = keras.models.load_model(
        model_name,
        custom_objects=custom_objects
    )

    UNET_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss=sm.losses.bce_dice_loss,
                       metrics=[model_metrics.my_iou_metric])

    test_sample_paths_com = model_pre_processing.load_data_paths([negative_sample[0], positive_sample[0]])
    test_com_image_data = model_pre_processing.load_test_dataset(
        test_sample_paths_com)  # apply pre-processing for resnet 50

    test_com_image_data = test_com_image_data.batch(batch_size)  # same as training
    results_com = UNET_model.evaluate(x=test_com_image_data, batch_size=batch_size)

    test_sample_paths_neg = model_pre_processing.load_data_paths(negative_sample)
    test_neg_image_data = model_pre_processing.load_test_dataset(
        test_sample_paths_neg)  # apply pre-processing for resnet 50

    test_neg_image_data = test_neg_image_data.batch(batch_size)  # same as training
    results_neg = UNET_model.evaluate(x=test_neg_image_data, batch_size=batch_size)

    test_sample_paths_pos = model_pre_processing.load_data_paths(positive_sample)
    test_pos_image_data = model_pre_processing.load_test_dataset(
        test_sample_paths_pos)  # apply pre-processing for resnet 50

    test_pos_image_data = test_pos_image_data.batch(batch_size)  # same as training
    results_pos = UNET_model.evaluate(x=test_pos_image_data, batch_size=batch_size)

    print(f"testing: {str(model_name)} \n\n")
    print(f"Non Dirt Road Sample \n neg_loss: {str(results_neg[0])} \n neg_IOU_score: {str(results_neg[1])}")
    print(f"Dirt Road Sample \n poss_loss: {str(results_pos[0])} \n poss_IOU_score: {str(results_pos[1])}")
    print(f"Combined Samples \n com_loss: {str(results_com[0])} \n com_IOU_score: {str(results_com[1])}")
    print("\n\n Test Finished")

