from typing import List, Any

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

from models.BaseModels.DataProvider import DataProvider
from matplotlib import pyplot as plt



class SequentialModel(Model, DataProvider):

    sequential_layers: List[keras.layers.Layer]

    def __init__(self, nmb_classes, optimizer, loss, metrics, MODEL_NAME=""):
        super(SequentialModel, self).__init__()
        self.register_data_provider(MODEL_NAME, nmb_classes)
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.sequential_layers = []


    def call(self, input, get_raw=False, **kwargs):
        result = input
        for layer in self.sequential_layers[:-1]:
            result = layer(result)
        if get_raw:
            activation_function = self.sequential_layers[-1].activation
            self.sequential_layers[-1].activation = lambda x: x
            result = self.sequential_layers[-1](result)
            self.sequential_layers[-1].activation = activation_function
        else:
            result = self.sequential_layers[-1](result)
        return result

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

    def train(self, epochs=1, train_data=None, callbacks=[]):
        tsboard = TensorBoard(log_dir=self.get_tensorboard_path(), histogram_freq=10, write_graph=True)
        checkpoint = ModelCheckpoint(self.SAVE_DIR, save_best_only=True, save_weights_only=True)
        lr_callback = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1, min_delta=0.005)
        callback = [tsboard, lr_callback, checkpoint] + callbacks
        train = self.train_dataset if train_data is None else train_data
        self.fit(train.repeat(),
                 epochs=epochs,
                 steps_per_epoch=self.train_steps_per_epoch,
                 callbacks=callback,
                 validation_data=self.test_dataset,
                 validation_steps=self.test_steps)

        self.evaluate(self.test_dataset, steps=self.test_steps)
        self.save_model_data()

    def get_dataset(self, split, name='', batch_size=32, shuffle=10000, nmb_classes=10):
        dataset =  super().get_dataset(split, name, batch_size, shuffle, nmb_classes)
        return dataset

    def test(self, test_data=None):
        test = self.test_dataset if test_data is None else test_data
        self.evaluate(test, steps=self.test_steps)


