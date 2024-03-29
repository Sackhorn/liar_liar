from tensorflow.python.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from liar_liar.models.base_models import MNIST_CONV_NAME
from liar_liar.models.mnist_models.mnist_model_base import MNISTModel


class MNISTConvModel(MNISTModel):

    def __init__(self, optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME=MNIST_CONV_NAME)
        self.sequential_layers = [
            Conv2D(32, [4, 4], activation='relu'),
            Dropout(0.1),
            MaxPool2D(),

            Conv2D(64, [3, 3], activation='relu'),
            Dropout(0.3),
            MaxPool2D(),

            Conv2D(128, [3, 3], activation='relu'),
            Dropout(0.5),
            MaxPool2D(),

            Flatten(),
            Dense(10, activation='softmax')]

if __name__ == "__main__":
    model = MNISTConvModel()
    model.train(epochs=5)
