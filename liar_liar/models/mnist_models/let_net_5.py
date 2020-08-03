from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, AveragePooling2D
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy

from liar_liar.models.base_models.model_names import LE_NET_NAME
from liar_liar.models.mnist_models.mnist_model_base import MNISTModel


class LeNet5(MNISTModel):

    def __init__(self, optimizer="sgd", loss=categorical_crossentropy, metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME=LE_NET_NAME)
        self.sequential_layers = [
            Conv2D(6, [5, 5], activation="tanh"),
            AveragePooling2D(strides=1),
            Conv2D(16, [5, 5], activation="tanh"),
            AveragePooling2D(),
            Conv2D(120, [5, 5], activation="tanh"),
            Flatten(),
            Dense(84, activation="tanh"),
            Dense(10, activation='softmax')
        ]


if __name__ == "__main__":
    model = LeNet5()
    model.train(epochs=15, augment_data=False)
