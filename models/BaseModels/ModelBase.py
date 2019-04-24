import os
from os.path import *
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt





class ModelBase(tf.keras.Model):
    ROOT_DIR = join(dirname(__file__), os.pardir)
    MODEL_DIR_NAME = "saved_models"

    def __init__(self, MODEL_NAME=""):
        super(ModelBase, self).__init__()
        if MODEL_NAME == "":
            raise NotImplementedError("Please support model name in subclass super call to this base class")
        self.SAVE_DIR = join(ModelBase.ROOT_DIR, ModelBase.MODEL_DIR_NAME, MODEL_NAME)

    def call(self, input):
        raise NotImplementedError("Implement call when overriding ModelBase")

    def get_dataset(self, split, name='', batch_size=32, shuffle=10000):
        def cast_labels(x, y):
            x = tf.cast(x, tf.float32)
            y = tf.one_hot(y, 10)
            return x, y

        dataset, info = tfds.load(name, split=split, with_info=True, as_supervised=True)  # type: tf.data.Dataset
        dataset = dataset.map(cast_labels).shuffle(shuffle).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        self.info = info
        self.batch_size = batch_size
        self.train_steps_per_epoch = info.splits['train'].num_examples // batch_size
        self.test_steps = info.splits['test'].num_examples // batch_size
        return dataset

    def test(self, test_data=None):
        raise NotImplementedError("Implement test when overriding ModelBase")

    def train(self, epochs=1, train_data=None):
        raise NotImplementedError()

    def load_model_data(self):
        self.load_weights(self.SAVE_DIR)
        print("Successfully loaded model from file: " + self.SAVE_DIR)
        self.test()

    def save_model_data(self):
        self.save_weights(self.SAVE_DIR)
        print("Successfully saved model data to: " + self.SAVE_DIR)
