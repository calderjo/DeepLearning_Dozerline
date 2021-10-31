import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras import metrics
import numpy as np
from unet_model import unet_classifier
from createDataSplit import create_data_split


def train_unet():
    """"
    source_raster = [w, x, y, z,]
    source_class = sc
    destination_folder = df
    create_dataset_method1(source, source_class, df)
    """

    # Creates a 70, 20, 10 percent split
    train_data, train_label, val_data, val_label, test_data, test_label = \
        create_data_split(image_directory="C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset/images/",
                          label_directory="C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset/labels/",
                          train_ratio=.7,
                          val_ratio=.2,
                          test_ratio=.1,
                          seed=479)

    model = unet_classifier()

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=metrics.Accuracy())

    model1_tb_callback = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True,
                                     write_steps_per_second=False, update_freq='epoch', profile_batch=2,
                                     embeddings_freq=0, embeddings_metadata=None)

    model.fit(x=train_data, y=train_label, batch_size=8, callbacks=[model1_tb_callback], epochs=8,
              validation_data=(val_data, val_label)
              )

    model.save("./Model_1")
    return


train_unet()
