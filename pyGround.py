import dataset_functions


def play_with_data():

    source_directory = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset"
    destination_directory = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset_t1"

    dataset_functions.create_training_validation(source_directory, destination_directory, .20, 479)


play_with_data()
