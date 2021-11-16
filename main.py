import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.data import AUTOTUNE

import segmentation_models as sm

import dataset_functions


def main():
    seed = 479  # seed for reproducibility
    input_size = (512, 512, 3)  # image size
    n_classes = 1  # one class for dozerline, non dozerline does not count as a class

    # this dir holds images that will be used for the training process
    training_img_path = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset_dozer_line/train"

    # here we create the train-val split 80 - 20
    # folder for resulting dir
    dataset_path = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset/"
    dataset_functions.create_training_validation(training_img_path, dataset_path, .20, seed)
    dataset = dataset_functions.load_training_validation_dataset(dataset_path, seed)

    BATCH_SIZE = 8
    BUFFER_SIZE = 700

    # -- Train Dataset --#
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=seed)
    dataset['train'] = dataset['train'].repeat(count=-1)
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    # -- Validation Dataset --#
    dataset['val'] = dataset['val'].repeat(count=-1)
    dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    # very cool, let's us visualize the training process
    s_unet_tensorflow = TensorBoard(log_dir='logs_unetM5',
                                    histogram_freq=0,
                                    write_graph=True,
                                    write_images=True,
                                    write_steps_per_second=False,
                                    update_freq='epoch',
                                    profile_batch=2,
                                    embeddings_freq=0,
                                    embeddings_metadata=None)

    sm.set_framework('tf.keras')

    model = sm.Unet(backbone_name='resnet50',
                    input_shape=input_size,
                    classes=n_classes,
                    activation='sigmoid',
                    encoder_weights='imagenet')

    print(model.summary())

    # here we set the opt and loss, metric values are printed during process
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=sm.losses.bce_jaccard_loss,
                  metrics=[sm.metrics.iou_score])

    EPOCHS = 20
    STEPS_PER_EPOCH = 616 // BATCH_SIZE
    VALIDATION_STEPS = 154 // BATCH_SIZE

    model.fit(x=dataset['train'],
              batch_size=4,
              callbacks=[s_unet_tensorflow],
              epochs=EPOCHS,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_steps=VALIDATION_STEPS,
              validation_data=dataset['val'])

    model.save("./unet_tensorflow_method5")
    model.save_weights("./unet_tensorflow_weights_method5")

    return 0


main()
