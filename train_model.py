import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.data import AUTOTUNE
import segmentation_models as sm
import dataset_functions


def unet_model_resnet_50_backbone(seed,
                                  training_set_path,
                                  batch_size,
                                  learning_rate,
                                  num_epochs,
                                  reshuffle_each_iteration,
                                  freeze_encoder,
                                  saving_path
                                  ):
    input_size = (512, 512, 3)
    n_classes = 1  # one class for dozerline, non dozerline does not count as a class
    # this dir holds images that will be used for the training process

    # here we create the train-val split 80 - 20
    dataset_path = saving_path[0]
    dataset_functions.create_training_validation(training_set_path, dataset_path, .20, seed)
    dataset = dataset_functions.load_training_validation_dataset(dataset_path, seed)

    buffer_size = 700

    # -- Train Dataset --#
    if reshuffle_each_iteration:
        dataset['train'] = dataset['train'].shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    else:
        dataset['train'] = dataset['train'].shuffle(buffer_size=buffer_size, seed=seed)

    dataset['train'] = dataset['train'].repeat(count=-1)
    dataset['train'] = dataset['train'].batch(batch_size)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    # -- Validation Dataset --#
    dataset['val'] = dataset['val'].repeat(count=-1)
    dataset['val'] = dataset['val'].batch(batch_size)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    # very cool, let's us visualize the training process
    s_unet_tensorflow = TensorBoard(log_dir=saving_path[1],
                                    histogram_freq=1,
                                    write_graph=True,
                                    write_images=True,
                                    write_steps_per_second=False,
                                    update_freq='epoch',
                                    profile_batch=2,
                                    embeddings_freq=0,
                                    embeddings_metadata=None)

    sm.set_framework('tf.keras')

    if freeze_encoder:
        model = sm.Unet(backbone_name='resnet50',
                        input_shape=input_size,
                        classes=n_classes,
                        activation='sigmoid',
                        encoder_freeze=True,
                        encoder_weights='imagenet')
    else:
        model = sm.Unet(backbone_name='resnet50',
                        input_shape=input_size,
                        classes=n_classes,
                        activation='sigmoid',
                        encoder_freeze=False,
                        encoder_weights='imagenet')

    print(model.summary())

    # here we set the opt and loss, metric values are printed during process
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=sm.losses.bce_jaccard_loss,
                  metrics=[sm.metrics.iou_score])

    steps_per_epoch = 616 // batch_size
    validation_steps = 154 // batch_size

    model.fit(x=dataset['train'],
              batch_size=batch_size,
              callbacks=[s_unet_tensorflow],
              epochs=num_epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              validation_data=dataset['val'])

    model.save(saving_path[2])
    model.save_weights(saving_path[3])

    return 0
