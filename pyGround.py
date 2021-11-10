import dataset_functions
import tensorflow as tf


def play_with_data():

    source_directory = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset/train"

    dataset_functions.training_images_transformation(source_directory)


play_with_data()



"""
   #  transformed_image = data_augmentation(image, mask)
    transformed_image_tensor = tf.convert_to_tensor(transformed_image['image'])
    transformed_mask_tensor = tf.convert_to_tensor(transformed_image['mask'])
    input_image, input_mask = normalize(transformed_image_tensor, transformed_mask_tensor)
      data_augmentation = data_aug.Compose([
        data_aug.HorizontalFlip(p=0.5),
        data_aug.VerticalFlip(p=0.5),
        data_aug.RandomBrightnessContrast(p=0.2)
    ])
    train_dataset = train_dataset.map(lambda image:
                                      training_set_functions(datapoint=image, data_augmentation=data_augmentation))                                   
"""