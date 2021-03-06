import os
import model_training
import tensorflow as tf
import segmentation_models as sm
import dataset_paths


def main():

    print(tf.config.list_physical_devices('GPU'))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    save_model_dir = "/home/jchavez/model/dozerline_extractor/unet/test"
    experiment_name = "test"

    try:
        os.makedirs(os.path.join(save_model_dir, experiment_name))
    except FileExistsError:
        pass

    constants_parameters = {
        'training_folder': [
            dataset_paths.pocket_imper_lidar["positive"],
            dataset_paths.south_nunns_imper_lidar["positive"],
            dataset_paths.south_tubbs_imper_lidar["positive"],
            dataset_paths.north_nunns_imper_lidar["positive"]
        ],
        'seed': 479
    }

    testing_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    trial_num = 0

    for numEpochs in testing_epochs:

        UNET_param = {
            'input_shape': (256, 256, 3),
            'batch_size': 32,
            'backbone_name': 'resnet18',
            'activation': 'sigmoid',
            'classes': 1,
            'loss': sm.losses.bce_dice_loss,
            'epochs': numEpochs,
            'learning_rate': .001
        }

        model_training.train_UNET_model(
            seed=constants_parameters['seed'],
            training_dirs=constants_parameters['training_folder'],
            model_params=UNET_param,
            experiment_target_dir=os.path.join(save_model_dir, experiment_name),
            trial_name=trial_num
        )

        trial_num += 1

    return 0


main()
