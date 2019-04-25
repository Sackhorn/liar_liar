import os

import tensorflow as tf
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D, BatchNormalization
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.regularizers import l2

from models.BaseModels.DataProvider import DataProvider
from models.CIFAR100Models.CIFAR100Model import CIFAR100Model


class ConvModel(CIFAR100Model):

    def __init__(self,
                 optimizer=Adam(),
                 loss=categorical_crossentropy,
                 metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="cifar100_conv_model")
        self.sequential_layers = [
            Conv2D(64, [3, 3], padding='same', activation='elu', kernel_regularizer=l2(), input_shape=(32,32,3)),
            BatchNormalization(),
            Conv2D(64, [3, 3], padding='same', activation='elu', kernel_regularizer=l2()),
            BatchNormalization(),
            MaxPool2D(),
            Dropout(rate=0.3),

            Conv2D(128, [3, 3], padding='same', activation='elu', kernel_regularizer=l2()),
            BatchNormalization(),
            Conv2D(128, [3, 3], padding='same', activation='elu', kernel_regularizer=l2()),
            BatchNormalization(),
            MaxPool2D(),
            Dropout(rate=0.4),

            Conv2D(256, [3, 3], padding='same', activation='elu', kernel_regularizer=l2()),
            BatchNormalization(),
            Conv2D(256, [3, 3], padding='same', activation='elu', kernel_regularizer=l2()),
            BatchNormalization(),
            MaxPool2D(),
            Dropout(rate=0.5),

            Flatten(),
            Dense(100, activation='softmax')
        ]

if __name__ == "__main__":
    enable_eager_execution()

    def lr_schedule(epoch):
        lrate = 0.001
        if epoch > 50:
            lrate = 0.0005
        if epoch > 100:
            lrate = 0.00025
        if epoch > 150:
            lrate = 0.000125
        if epoch > 200:
            lrate = 0.000062
        if epoch > 250:
            lrate = 0.000031
        if epoch > 300:
            lrate = 0.000015
        return lrate

    model = ConvModel()
    lr_callback = LearningRateScheduler(lr_schedule)
    model.train(epochs=350, callbacks=[lr_callback])
