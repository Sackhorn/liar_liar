from math import floor

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from models.CIFAR10Models.CIFAR10Model import CIFAR10Model


class ConvModel(CIFAR10Model):

    def __init__(self,
                 optimizer=tf.train.AdamOptimizer(),
                 loss=tf.keras.losses.categorical_crossentropy,
                 metrics=['accuracy']):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="cifar10_conv_model")
        self.sequential_layers = [
            tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, [3, 3], activation='relu'),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(256, [3, 3], activation='relu'),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Conv2D(256, [3, 3], activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ]

if __name__ == "__main__":
    tf.enable_eager_execution()
    model = ConvModel()
    model.train(epochs=20)
