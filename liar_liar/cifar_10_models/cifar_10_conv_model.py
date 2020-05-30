import tensorflow as tf
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, MaxPool2D, BatchNormalization, Conv2D
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from liar_liar.base_models.model_names import CIFAR_10_CONV_NAME
from liar_liar.cifar_10_models.cifar_10_model_base import CIFAR10Model


class CIFAR10ConvModel(CIFAR10Model):

    def __init__(self, optimizer=Adam(), loss=categorical_crossentropy, metrics=['categorical_accuracy', 'top_k_categorical_accuracy']):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME=CIFAR_10_CONV_NAME)
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
    model = CIFAR10ConvModel()
    model.train(epochs=150)
