import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard

from keras_unet import models

from unet_model import unet_classifier


def train_unet(train, val, model_name):

    model = models.custom_unet(input_shape=(512, 512, 3))
    print(model.summary())




    """
        model = unet_classifier((572, 572, 3))

    print(model.summary())
    
    

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics={keras.metrics.Accuracy(), tf.keras.metrics.MeanIoU(num_classes=2)}
                  )

    model1_tb_callback = TensorBoard(log_dir='logs',
                                     histogram_freq=0,
                                     write_graph=True,
                                     write_images=True,
                                     write_steps_per_second=False,
                                     update_freq='epoch',
                                     profile_batch=2,
                                     embeddings_freq=0,
                                     embeddings_metadata=None
                                     )

    model.fit(x=train,
              batch_size=8,
              callbacks=[model1_tb_callback],
              epochs=10,
              validation_data=val
              )

    model.save("./Model_1")
    
    """

    return 0

