from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from models.MNISTModels.MNISTModel import MNISTModel


class DenseModel(MNISTModel):

    def __init__(self, optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="mnist_dense_model")
        self.sequential_layers = [
            Flatten(),
            Dense(784, activation='sigmoid'),
            Dense(800, activation='sigmoid'),
            Dense(800, activation='sigmoid'),
            Dense(10, activation='softmax')
        ]


if __name__ == "__main__":
    enable_eager_execution()
    model = DenseModel()
    model.train(epochs=50)
