import tensorflow as tf
from models.MNISTModels.MNISTModel import MNISTModel


class DenseModel(MNISTModel):

    def __init__(self,
                 optimizer=tf.train.AdamOptimizer(),
                 loss=tf.keras.losses.categorical_crossentropy,
                 metrics=['accuracy']):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="mnist_dense_model")
        self.sequential_layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(784, activation='sigmoid'),
            tf.keras.layers.Dense(800, activation='sigmoid'),
            tf.keras.layers.Dense(800, activation='sigmoid'),
            tf.keras.layers.Dense(10, activation='softmax')
        ]


if __name__ == "__main__":
    tf.enable_eager_execution()
    model = DenseModel()
    model.train()