import tensorflow as tf
import model_pre_processing
import segmentation_models as sm
import model_metrics
import numpy as np
from PIL import Image
from tensorflow import keras
from sklearn import metrics

def map_wide_based_evaluation(prediction_map, ground_truth_map):
    Image.MAX_IMAGE_PIXELS = 2000000000

    prediction_array = Image.open(prediction_map)
    prediction_array = np.array(prediction_array)

    ground_truth_array = Image.open(ground_truth_map)
    ground_truth_array = np.array(ground_truth_array)

    print(f'IOU score (map wide): {model_metrics.my_iou_metric(ground_truth_array, prediction_array)}')

    ground_truth_array = ground_truth_array.flatten()
    prediction_array = prediction_array.flatten()

    print(f'F1 score weighted (map wide): {metrics.f1_score(ground_truth_array, prediction_array, average="weighted", zero_division=1)}')
    print(f'Precision score weighted (map wide): {metrics.precision_score(ground_truth_array, prediction_array, average="weighted", zero_division=1)}')
    print(f'Recall score weighted (map wide): {metrics.recall_score(ground_truth_array, prediction_array, average="weighted", zero_division=1)}')
    print(f'Accuracy score weighted (map wide): {metrics.accuracy_score(ground_truth_array, prediction_array)}')


def image_chip_based_evaluation(model_name, custom_objects, positive_sample, negative_sample, batch_size):

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

