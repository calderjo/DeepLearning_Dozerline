import createDataset
import createDataSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from keras_unet import models
from arcpy import sa


def main():
    # this dir holds images that will be used for the training process
    training_img_path = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset/train"

    # this dir holds images that will be with held from the training process
    test_img_path = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset/test"

    input_size = (512, 512, 3)  # image size
    n_classes = 2
    seed = 479  # seed for reproducibility

    # here we create the train-val split 80 - 20
    # folder for resulting dir
    dataset_path = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset/"
    createDataSplit.create_training_validation(training_img_path, dataset_path, .20, seed)
    dataset = createDataSplit.load_image_dataset(dataset_path, seed)

    model = models.satellite_unet(input_shape=input_size, num_classes=n_classes)

    print(model.summary())

    # here we set the opt and loss, metric values are printed during process
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics={keras.metrics.Accuracy(), tf.keras.metrics.MeanIoU(num_classes=2)})

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
              batch_size=8,
              callbacks=[s_unet_tensorflow],
              epochs=10,
              validation_data=dataset['val'])

    model.save("s_unet_tensorflow")
    model.save_weights("./s_unet_tensorflow_weights")

    return 0


main()
