import tensorflow as tf
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, MaxPool2D, BatchNormalization, Conv2D
from tensorflow.python.keras.optimizer_v2.adam import Adam

from models.CIFAR10Models.CIFAR10Model import CIFAR10Model


class ConvModel(CIFAR10Model):

    def __init__(self,
                 optimizer=Adam(),
                 loss=tf.keras.losses.categorical_crossentropy,
                 metrics=['categorical_accuracy']):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="cifar10_conv_model")
        self.sequential_layers = [
            Conv2D(32, [3, 3], padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(32,32,3)),
            BatchNormalization(),
            Conv2D(32, [3, 3], padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            BatchNormalization(),
            MaxPool2D(),
            Dropout(rate=0.2),

            Conv2D(64, [3, 3], padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            BatchNormalization(),
            Conv2D(64, [3, 3], padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            BatchNormalization(),
            MaxPool2D(),
            Dropout(rate=0.3),

            Conv2D(128, [3, 3], padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            BatchNormalization(),
            Conv2D(128, [3, 3], padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            BatchNormalization(),
            MaxPool2D(),
            Dropout(rate=0.4),

            Flatten(),
            Dense(10, activation='softmax')
        ]

if __name__ == "__main__":
    enable_eager_execution()
    model = ConvModel()
    model.train(epochs=150)
