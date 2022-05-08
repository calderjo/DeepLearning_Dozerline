import os
import train_model
import tensorflow as tf


def main():
    print(tf.config.list_physical_devices('GPU'))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    UNET_param = {
        'batch_size': 32,
        'epochs': 0,
        'input_size': (256, 256, 3),
        'learning_rate': 1e-2,
        'backbone': 'resnet50'
    }

    training_base_dir = "/home/jchavez/dataset/Dirt_Gravel_Roads_False_Image"
    save_model_dir = "/home/jchavez/model/dozerline_extractor/unet/test"
    experiment_name = "test2"

    try:
        os.makedirs(os.path.join(save_model_dir, experiment_name))
    except FileExistsError:
        pass

    constants_parameters = {
        'training_folder': [
            os.path.join(training_base_dir, "Pocket", "positive_samples"),
            os.path.join(training_base_dir, "South_Nunns", "positive_samples"),
            os.path.join(training_base_dir, "North_Nunns", "positive_samples"),
            os.path.join(training_base_dir, "South_Tubbs", "positive_samples")
        ],
        'seed': 479
    }

    testing_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    trial_num = 0

    for numEpochs in testing_epochs:
        UNET_param['epochs'] = numEpochs
        train_model.train_UNET_RESNET50_model(seed=constants_parameters['seed'],
                                              training_dirs=constants_parameters['training_folder'],
                                              unet_params=UNET_param,
                                              experiment_target_dir=os.path.join(save_model_dir, experiment_name),
                                              trial_number=trial_num)
        trial_num += 1

    return 0


main()
