from math import floor

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.training.adam import AdamOptimizer

from models.CIFAR10Models.CIFAR10Model import CIFAR10Model


class ConvModel(CIFAR10Model):

    def __init__(self,
                 optimizer=Adam(decay=1e-6),
                 loss=tf.keras.losses.categorical_crossentropy,
                 metrics=[tf.keras.metrics.categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="cifar10_conv_model")
        self.sequential_layers = [
            tf.keras.layers.Conv2D(32, [3, 3], padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(32,32,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, [3, 3], padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(rate=0.2),

            tf.keras.layers.Conv2D(64, [3, 3], padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, [3, 3], padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(rate=0.3),

            tf.keras.layers.Conv2D(128, [3, 3], padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, [3, 3], padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(rate=0.4),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ]

if __name__ == "__main__":
    enable_eager_execution()

    def lr_schedule(epoch):
        lrate = 0.001
        if epoch > 75:
            lrate = 0.0005
        if epoch > 100:
            lrate = 0.0003
        return lrate

    model = ConvModel()
    w_scheulde = LearningRateScheduler(lr_schedule)
    model.train(epochs=150, callbacks=[w_scheulde])
