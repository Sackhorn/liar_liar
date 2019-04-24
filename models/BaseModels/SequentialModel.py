import tensorflow as tf
from tensorflow.python.keras import Model
from models.BaseModels.DataProvider import DataProvider
from matplotlib import pyplot as plt



class SequentialModel(Model, DataProvider):

    def __init__(self, nmb_classes, optimizer, loss, metrics, MODEL_NAME=""):
        super(SequentialModel, self).__init__()
        self.register_data_provider(MODEL_NAME, nmb_classes)
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    def call(self, input):
        result = self.sequential_layers[0](input)
        for layer in self.sequential_layers[1:]:
            result = layer(result)
        return result

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

    def train(self, epochs=1, train_data=None, callbacks=None):
        class LossHistory(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.losses = []

            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))


        history = LossHistory()
        callback = [history]
        if callbacks is not None: callback.append(callbacks)
        train = self.train_dataset if train_data is None else train_data
        self.fit(train.repeat(),
                 epochs=epochs,
                 steps_per_epoch=self.train_steps_per_epoch,
                 callbacks=callback,
                 validation_data=self.train_dataset,
                 validation_steps=self.test_steps)

        self.evaluate(self.test_dataset,  steps=self.test_steps)
        self.evaluate(self.train_dataset,  steps=self.train_steps_per_epoch)
        self.save_model_data()
        plt.plot(history.losses)
        plt.show()

    def get_dataset(self, split, name='', batch_size=32, shuffle=10000, nmb_classes=10):
        dataset =  super().get_dataset(split, name, batch_size, shuffle, nmb_classes)
        dataset = dataset.map(lambda x,y: (tf.divide(x, 255), y))
        return dataset

    def test(self, test_data=None):
        test = self.test_dataset if test_data is None else test_data
        self.evaluate(test, steps=self.test_steps)


