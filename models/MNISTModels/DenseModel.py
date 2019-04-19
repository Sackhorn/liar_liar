import tensorflow as tf
from models.MNISTModels.MNISTModel import MNISTModel


class DenseModel(MNISTModel):

    def __init__(self):
        super(DenseModel, self).__init__(MODEL_NAME="mnist_dense_model")
        self.sequential_layers = [
            tf.keras.layers.Dense(784, activation='sigmoid'),
            tf.keras.layers.Dense(800, activation='sigmoid'),
            tf.keras.layers.Dense(800, activation='sigmoid'),
            tf.keras.layers.Dense(10)
        ]


    def call(self, input):
        input = tf.reshape(input, [-1,784])
        return super().call(input)


if __name__ == "__main__":
    tf.enable_eager_execution()
    model = DenseModel()
    model.train()
