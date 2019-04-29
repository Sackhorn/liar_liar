import tensorflow as tf
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.optimizer_v2.adam import Adam

from models.MNISTModels.MNISTModel import MNISTModel


class ConvModel(MNISTModel):

    def __init__(self,
                 optimizer=Adam(),
                 loss=tf.keras.losses.categorical_crossentropy,
                 metrics=['categorical_accuracy']):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="mnist_conv_model")
        self.sequential_layers = [
            tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(128, [3, 3], activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')]

if __name__ == "__main__":
    enable_eager_execution()
    model = ConvModel()
    model.train(epochs=5)
