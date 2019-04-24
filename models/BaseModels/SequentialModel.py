from models.BaseModels.ModelBase import ModelBase
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt



class SequentialModel(ModelBase):

    def __init__(self, nmb_classes, optimizer, loss, metrics, MODEL_NAME=""):
        super(SequentialModel, self).__init__(MODEL_NAME=MODEL_NAME)
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.train_dataset = self.get_dataset(tfds.Split.TRAIN)
        self.test_dataset = self.get_dataset(tfds.Split.TEST)

    def call(self, input):
        result = self.sequential_layers[0](input)
        for layer in self.sequential_layers[1:]:
            result = layer(result)
        return result

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

    def train(self, epochs=1, train_data=None):
        class LossHistory(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.losses = []

            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))
        history = LossHistory()
        train = self.train_dataset if train_data is None else train_data
        self.fit_generator(train.repeat(),
                           epochs=epochs,
                           steps_per_epoch=self.train_steps_per_epoch,
                           callbacks=[history])
        self.evaluate(self.test_dataset, batch_size=self.batch_size, steps=self.test_steps)
        # self.evaluate(self.train_dataset, batch_size=self.batch_size, steps=self.train_steps_per_epoch)
        self.save_model_data()
        plt.plot(history.losses)
        plt.show()

    def test(self, test_data=None):
        test = self.test_dataset if test_data is None else test_data
        self.evaluate(self.test_dataset, batch_size=self.batch_size, steps=self.test_steps)


