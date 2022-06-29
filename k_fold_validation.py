import constant_values
import train_model

UNET_param = {
    'batch_size': 32,
    'epochs': 15,
    'input_size': (256, 256, 3),
    'learning_rate': 0.01,
    'backbone': 'resnet18'
}

folds = [
    constant_values.north_nunns_imper_lidar["positive"],
    constant_values.north_tubbs_imper_lidar["positive"],
    constant_values.south_nunns_imper_lidar["positive"],
    constant_values.south_tubbs_imper_lidar["positive"],
    constant_values.pocket_imper_lidar["positive"]
]

fold1_training = [x for i, x in enumerate(folds) if i != 0]
fold2_training = [x for i, x in enumerate(folds) if i != 1]
fold3_training = [x for i, x in enumerate(folds) if i != 2]
fold4_training = [x for i, x in enumerate(folds) if i != 3]
fold5_training = [x for i, x in enumerate(folds) if i != 4]

training_folds = [fold1_training, fold2_training, fold3_training, fold4_training, fold5_training]

count = 3
for fold in training_folds[2:]:
    train_model.train_UNET_RESNET_model(
        seed=479,
        training_dirs=fold,
        unet_params=UNET_param,
        experiment_target_dir="/home/jchavez/model/dozerline_extractor/unet/lofo/experiment2",
        trial_number=count
    )
    count += 1
