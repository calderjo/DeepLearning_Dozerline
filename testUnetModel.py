import tensorflow as tf
from tensorflow import keras
import dataset_functions
import segmentation_models as sm


def test_unet(model_name, test_img, seed):
    test_img_path = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset_dozer_line/test"
    test_set = dataset_functions.load_test_dataset(test_img_path, seed)
    test_set.batch(8)

    model = keras.models.load_model('path/to/location')

    if model:
        for image, mask in model.take(10):
            pred_mask = model.predict(image)
            # dataset_functions.display_sample([image[0], mask, create_mask(pred_mask)])


def evaluate_model(model_path, test_data_path):
    return


"""
def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    Return a filter mask with the top 1 predictions
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
        for each pixels.
    
    # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask
"""
