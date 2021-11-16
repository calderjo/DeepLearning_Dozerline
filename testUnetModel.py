# %%
import os
import tensorflow as tf
from tensorflow import keras
import dataset_functions
import segmentation_models as sm


def evaluate_model(model_name, test_set_path, batch_size):

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

    visualize_predictions(test_image_data, unet_model, test_set_path)  # viz predictions made by model


def visualize_predictions(test_image_data, model_path, test_set_path):

    # finds all images in the test set
    files = os.listdir(os.path.join(test_set_path, "images"))
    test_images = []
    for file in files:
        if file.endswith(".png"):
            test_images.append(file)

    count = 0
    for image, mask in test_image_data.take(4):   # for all the images in test set
        predictions = model_path.predict(image).round()   # make a prediction

        for i in range(0, 8):  # plot prediction with the input image and ground truth
            image = tf.io.read_file((os.path.join(test_set_path, "images", test_images[count])))
            image = tf.image.decode_png(image, channels=3, dtype=tf.uint16)
            image = tf.image.convert_image_dtype(image, tf.uint8)
            dataset_functions.display_sample([image, mask[i], predictions[i]])
            count += 1


evaluate_model(
    model_name="C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/unet_tensorflow_method5",
    test_set_path="C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset_dozer_line/test",
    batch_size=8
)

