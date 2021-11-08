from tensorflow.python.data import AUTOTUNE

import raster_to_image_chips
import dataset_functions
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import segmentation_models as sm

import dataset_functions


def main():

    # this dir holds images that will be used for the training process
    # this dir holds images that will be used for the training process
    training_img_path = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset_dozer_line/train"

    input_size = (512, 512, 3)  # image size
    n_classes = 1
    seed = 479  # seed for reproducibility

    # here we create the train-val split 80 - 20
    # folder for resulting dir
    dataset_path = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset/"
    dataset_functions.create_training_validation(training_img_path, dataset_path, .20, seed)
    dataset = dataset_functions.load_image_dataset(dataset_path, seed)

    BATCH_SIZE = 8
    BUFFER_SIZE = 1000

    # -- Train Dataset --#
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=seed)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    # -- Validation Dataset --#
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    for image, mask in dataset['train'].take(1):
        sample_image, sample_mask = image, mask

    dataset_functions.display_sample([sample_image[0], sample_mask[0]])


    """"
    print(dataset['train'])
    print(dataset['val'])

    EPOCHS = 1

    STEPS_PER_EPOCH = 616 // BATCH_SIZE
    VALIDATION_STEPS = 154 // BATCH_SIZE

    keras.backend.set_image_data_format('channels_last')
    # or keras.backend.set_image_data_format('channels_first')
    
    model = sm.Unet('resnet34',
                    classes=1,
                    activation='sigmoid',
                    input_size=(512, 512, 3),
                    encoder_weights='imagenet')


    print(model.summary())

    # here we set the opt and loss, metric values are printed during process
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=keras.metrics.Accuracy())

    # very cool, let's us visualize the training process
    s_unet_tensorflow = TensorBoard(log_dir='logs',
                                    histogram_freq=0,
                                    write_graph=True,
                                    write_images=True,
                                    write_steps_per_second=False,
                                    update_freq='epoch',
                                    profile_batch=2,
                                    embeddings_freq=0,
                                    embeddings_metadata=None)

    model.fit(x=dataset['train'],
              batch_size=4,
              callbacks=[s_unet_tensorflow],
              epochs=EPOCHS,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_steps=VALIDATION_STEPS,
              validation_data=dataset['val'])

    model.save("s_unet_tensorflow")
    model.save_weights("./s_unet_tensorflow_weights")
    return 0
    """


main()
