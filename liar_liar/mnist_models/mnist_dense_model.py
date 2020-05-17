from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from liar_liar.base_models.model_names import MNIST_DENSE_NAME
from liar_liar.mnist_models.mnist_model_base import MNISTModel

class MNISTDenseModel(MNISTModel):

    def __init__(self, optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME=MNIST_DENSE_NAME)
        self.sequential_layers = [
            Flatten(),
            Dense(784, activation='sigmoid'),
            Dense(800, activation='sigmoid'),
            Dense(800, activation='sigmoid'),
            Dense(10, activation='softmax')
        ]


if __name__ == "__main__":
    enable_eager_execution()
    model = MNISTDenseModel()
    model.train(epochs=5)
