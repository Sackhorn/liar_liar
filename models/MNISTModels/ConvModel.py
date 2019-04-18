import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from models.BaseModels.SequentialModel import SequentialModel
from models.MNISTModels.MNISTModel import MNISTModel


class ConvModel(MNISTModel):

    def __init__(self):
        super(ConvModel, self).__init__(MODEL_NAME="mnist_conv_model")

        self.conv1 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D()
        self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D()
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(100, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(10)
        self.sequential_layers = [self.conv1, self.conv2, self.pool1, self.conv3, self.pool2, self.dropout1, self.flatten1, self.dense1, self.dropout2, self.dense2]

if __name__ == "__main__":
    tf.enable_eager_execution()
    model = ConvModel()
    model.train()
