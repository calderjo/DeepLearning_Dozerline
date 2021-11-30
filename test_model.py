import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import dataset_functions
import segmentation_models as sm
from sklearn.metrics import jaccard_score


def evaluate_model(model_name, test_set_path, batch_size, visualize):

    test_image_data = dataset_functions.load_test_dataset(test_set_path)  # apply pre-processing for resnet 50
    test_image_data = test_image_data.batch(8)  # same as training

    unet_model = keras.models.load_model(model_name,
                                         custom_objects={
                                             'iou_score': sm.metrics.iou_score,
                                             'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss}
                                         )

    results = unet_model.evaluate(x=test_image_data, batch_size=batch_size)  # makes prediction on whole test set
    print("loss: " + str(results[0]))   # printing the loss and iou_score
    print("iou_score: " + str(results[1]))

    if visualize:
        visualize_predictions(test_image_data, unet_model, test_set_path)  # viz predictions made by model


def visualize_predictions(test_image_data, model_path, test_set_path):

    tf.executing_eagerly()

    # finds all images in the test set
    files = os.listdir(os.path.join(test_set_path, "images"))

    test_images = []
    for file in files:
        if file.endswith(".png"):
            test_images.append(file)
    test_images = sorted(test_images)

    count = 0
    for image, mask in test_image_data.take(4):   # for all the images in test set

        predictions = model_path.predict(image).round()  # make a prediction
        for i in range(0, 8):  # plot prediction with the input image and ground truth

            mask_val = np.ravel(mask[i].numpy(), order='C')
            pred_val = np.ravel(predictions[i], order='C')

            prediction_score = jaccard_score(mask_val, pred_val)

            input_image = tf.io.read_file((os.path.join(test_set_path, "images", test_images[count])))
            input_image = tf.image.decode_png(input_image, channels=3, dtype=tf.uint16)
            input_image = tf.image.convert_image_dtype(input_image, tf.uint8)

            dataset_functions.display_sample([input_image, mask[i], predictions[i]], prediction_score)
            count += 1


evaluate_model(
    model_name="C:/Users/jonat/Documents/deepLearningModel/dozerlineExtraction/model_v1/Models/test_freeze_encoding"
               "/unet_v2_method_0a",
    test_set_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/dataset_dozer_line/test",
    batch_size=8,
    visualize=True
)

