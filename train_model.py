import dataset_functions
import os
import segmentation_models as sm
import sklearn.model_selection
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.callbacks import TensorBoard
import iou_score_metric


def write_model_param_file(unet_params, experiment_dir, trial_number):
    fileName = f"fold_{trial_number}_params"
    param_file = open(fileName, mode="a")

    input_size = unet_params['input_size']
    batch_size = unet_params['batch_size']
    epochs = unet_params['epochs']
    learning_rate = unet_params['learning_rate']
    backbone = unet_params['backbone']

    param_file.write(f"input size: {str(input_size)} \n")
    param_file.write(f"batch size: {str(batch_size)} \n")
    param_file.write(f"epochs: {str(epochs)} \n")
    param_file.write(f"learning rate {str(learning_rate)} \n")
    param_file.write(f"backbone {str(backbone)} \n")
    param_file.close()

    return


def build_UNET_RESNET_model(learning_rate, input_size, backbone):

    model = sm.Unet(backbone_name=backbone,
                    input_shape=input_size,
                    classes=1,
                    activation='sigmoid',
                    encoder_freeze=True,
                    encoder_weights='imagenet')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=sm.losses.dice_loss,
        metrics=[iou_score_metric.my_iou_metric, sm.losses.dice_loss]
    )

    return model


def train_UNET_RESNET_model(
        seed,
        training_dirs,
        unet_params,
        experiment_target_dir,
        trial_number
):
    write_model_param_file(unet_params, experiment_target_dir, trial_number)
    data_samples = dataset_functions.load_data_paths(training_dirs)

    shuffle_split = sklearn.model_selection.ShuffleSplit(n_splits=1, test_size=.3, random_state=seed)
    split_gen = shuffle_split.split(X=data_samples)
    train_indexes, val_indexes = next(split_gen)

    dataset = dataset_functions.load_training_validation_dataset(
        training=data_samples[train_indexes],
        validation=data_samples[val_indexes],
        seed=seed
    )

    data_samples = None
    buffer_size = len(train_indexes)

    # Train Dataset prepare batches
    dataset['train'] = dataset['train'].shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    dataset['train'] = dataset['train'].repeat(count=-1)
    dataset['train'] = dataset['train'].batch(unet_params['batch_size'])
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    # Validation Dataset prepare batches
    dataset['val'] = dataset['val'].repeat(count=-1)
    dataset['val'] = dataset['val'].batch(unet_params['batch_size'])
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    steps_per_epoch = len(train_indexes) // unet_params['batch_size']
    validation_steps = len(val_indexes) // unet_params['batch_size']

    # very cool, let's us visualize the training process
    logger = TensorBoard(log_dir=os.path.join(experiment_target_dir, f"fold_{trial_number}_log"),
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=0,
                         embeddings_metadata=None)

    model = build_UNET_RESNET_model(
        input_size=unet_params['input_size'],
        learning_rate=unet_params['learning_rate'],
        backbone=unet_params['backbone'])

    model.fit(
        x=dataset['train'],
        batch_size=unet_params['batch_size'],
        epochs=unet_params['epochs'],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_data=dataset['val'],
        callbacks=[logger]
    )

    model.save(os.path.join(experiment_target_dir, f"fold_{str(trial_number)}_model"))
    return
