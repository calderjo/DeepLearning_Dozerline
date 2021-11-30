import train_model


def main():
    constants_parameters = {'training_path': "C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips"
                                             "/bands_IRG_dozer_line/train",
                            'seed': 479,
                            'batch_size': 8,
                            'learning_rate': 0.001,
                            'reshuffle': True,
                            'freeze_encoder': True}

    method = "bands_IRG_"

    epochs_to_test = [5, 10, 15, 20, 25]

    for i in range(0, 5):

        v = i+1
        version = str(v)+"_"
        saving_path = ["C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/train_val_split/",
                       "unet_v_" + "method_" + method + version + "logs",
                       "unet_v_" + "method_" + method + version,
                       "unet_v_" + "method_" + method + version + "weights"]

        print("Starting " + "unet_v_" + "method_" + method + version + " model Training")

        print("epoch " + str(epochs_to_test[i]))

        train_model.unet_model_resnet_50_backbone(seed=constants_parameters['seed'],
                                                  training_set_path=constants_parameters['training_path'],
                                                  batch_size=constants_parameters['batch_size'],
                                                  learning_rate=constants_parameters['learning_rate'],
                                                  num_epochs=epochs_to_test[i],
                                                  reshuffle_each_iteration=constants_parameters['reshuffle'],
                                                  saving_path=saving_path,
                                                  freeze_encoder=constants_parameters['freeze_encoder'])

        print("Finished " + "unet_v_" + "method_" + method + version + " model Training")

    return 0


main()
