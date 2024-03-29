from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D, BatchNormalization
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.regularizers import l2

from liar_liar.models.base_models import CIFAR_100_CONV_NAME
from liar_liar.models.cifar_100_models.cifar_100_model_base import CIFAR100Model


class CIFAR100ConvModel(CIFAR100Model):

    def __init__(self, optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy, 'top_k_categorical_accuracy']):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME=CIFAR_100_CONV_NAME)
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
    model = CIFAR100ConvModel()
    model.train(epochs=1000)
