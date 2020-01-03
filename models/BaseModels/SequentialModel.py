from typing import List
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow_datasets import Split
from models.BaseModels.DataProvider import DataProvider



class SequentialModel(Model, DataProvider):

    sequential_layers: List[keras.layers.Layer]

    def __init__(self, optimizer, loss, metrics, MODEL_NAME="", dataset_name='', dataset_dir=''):
        super(SequentialModel, self).__init__()
        self.register_data_provider(MODEL_NAME, dataset_name, dataset_dir)
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.sequential_layers = []

    @tf.function
    def call(self, input, training=False, get_raw=False, **kwargs):
        result = input
        for layer in self.sequential_layers[:-1]:
            result = layer(result, training=training)
        if get_raw:
            activation_function = self.sequential_layers[-1].activation
            self.sequential_layers[-1].activation = lambda x: x
            result = self.sequential_layers[-1](result)
            self.sequential_layers[-1].activation = activation_function
        else:
            result = self.sequential_layers[-1](result)
        return result

    def train(self, epochs=1, train_data=None, callbacks=[], augment_data=True):
        tsboard = TensorBoard(log_dir=self.get_tensorboard_path(), histogram_freq=10, write_graph=True)
        checkpoint = ModelCheckpoint(self.SAVE_DIR, save_best_only=True, save_weights_only=True)
        lr_callback = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1, min_delta=0.005)
        callback = [tsboard, lr_callback, checkpoint] + callbacks

        test = self.get_dataset(Split.TEST)
        train = self.get_dataset(Split.TRAIN, augment_data=augment_data) if train_data is None else train_data
        self.fit(train.repeat(),
                 epochs=epochs,
                 steps_per_epoch=self.train_steps_per_epoch,
                 callbacks=callback,
                 validation_data=test,
                 validation_steps=self.test_steps)

        self.evaluate(test, steps=self.test_steps)
        self.save_model_data()

    def test(self, test_data=None):
        test = self.get_dataset(Split.TEST) if test_data is None else test_data
        self.evaluate(test, steps=self.test_steps)


