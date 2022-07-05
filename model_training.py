import model_pre_processing
import os
import segmentation_models as sm
import sklearn.model_selection
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.callbacks import TensorBoard


def write_model_param_file(model_params, experiment_dir, trial_name):
    fileName = f"{trial_name}_params"
    param_file = open(fileName, mode="a")

    # writing setting to file
    param_file.write(f"input_shape: {str(model_params['input_shape'])} \n")
    param_file.write(f"batch_size: {str(model_params['batch_size'])} \n")
    param_file.write(f"backbone_name: {str(model_params['backbone_name'])} \n")
    param_file.write(f"activation: {str(model_params['activation'])} \n")
    param_file.write(f"classes: {str(model_params['classes'])} \n")
    param_file.write(f"loss: {str(model_params['loss'].name)} \n")
    param_file.write(f"epochs: {str(model_params['epochs'])} \n")
    param_file.write(f"learning_rate: {str(model_params['learning_rate'])} \n")

    param_file.close()
    os.rename(fileName, os.path.join(experiment_dir, fileName))
    return


def build_UNET_model(model_params):

    model = sm.Unet(backbone_name=model_params['backbone_name'],
                    input_shape=model_params['input_shape'],
                    classes=model_params['classes'],
                    activation=model_params['activation'],
                    encoder_freeze=True,
                    encoder_weights='imagenet')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_params['learning_rate']),
                  loss=model_params['loss'],
                  metrics=[sm.metrics.iou_score])
    return model


def train_UNET_model(
        seed,
        training_dirs,
        model_params,
        experiment_target_dir,
        trial_name
):
    write_model_param_file(model_params, experiment_target_dir, trial_name)
    data_samples = model_pre_processing.load_data_paths(training_dirs)

    shuffle_split = sklearn.model_selection.ShuffleSplit(n_splits=1, test_size=.3, random_state=seed)
    split_gen = shuffle_split.split(X=data_samples)
    train_indexes, val_indexes = next(split_gen)

    dataset = model_pre_processing.load_training_validation_dataset(
        training=data_samples[train_indexes],
        validation=data_samples[val_indexes],
        seed=seed
    )

    data_samples = None
    buffer_size = len(train_indexes)

    # Train Dataset prepare batches
    dataset['train'] = dataset['train'].shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    dataset['train'] = dataset['train'].repeat(count=-1)
    dataset['train'] = dataset['train'].batch(model_params['batch_size'])
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    # Validation Dataset prepare batches
    dataset['val'] = dataset['val'].repeat(count=-1)
    dataset['val'] = dataset['val'].batch(model_params['batch_size'])
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    steps_per_epoch = len(train_indexes) // model_params['batch_size']
    validation_steps = len(val_indexes) // model_params['batch_size']

    # very cool, let's us visualize the training process
    logger = TensorBoard(log_dir=os.path.join(experiment_target_dir, f"{trial_name}_log"),
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=0,
                         embeddings_metadata=None)

    model = build_UNET_model(model_params)

    model.fit(
        x=dataset['train'],
        batch_size=model_params['batch_size'],
        epochs=model_params['epochs'],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_data=dataset['val'],
        callbacks=[logger]
    )

    model.save(os.path.join(experiment_target_dir, f"{trial_name}_model"))
