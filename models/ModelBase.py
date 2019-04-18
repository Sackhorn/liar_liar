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

    def get_dataset(self, type, batch_size=32):
        raise NotImplementedError("Implement get_dataset when overriding ModelBase")

    def test(self, test_data=None):
        raise NotImplementedError("Implement test when overriding ModelBase")

    def train(self, train_data=None, optimizer=tf.train.AdamOptimizer()):
        raise NotImplementedError()

    def load_model_data(self):
        self.load_weights(self.SAVE_DIR)
        print("Successfully loaded model from file: " + self.SAVE_DIR)
        self.test()

    def save_model_data(self):
        self.save_weights(self.SAVE_DIR)
        print("Successfully saved model data to: " + self.SAVE_DIR)
