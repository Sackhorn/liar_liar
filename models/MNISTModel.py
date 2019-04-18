import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from os.path import *
from models.ModelBase import ModelBase

class MNISTModel(ModelBase):

    def __init__(self):
        super(MNISTModel, self).__init__(MODEL_NAME="mnist_model")
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
        self.loss_history = []

        self.train_dataset = self.get_dataset(tfds.Split.TRAIN)
        self.test_dataset = self.get_dataset(tfds.Split.TEST)

    def call(self, input):
        result = self.conv1(input)
        result = self.conv2(result)
        result = self.pool1(result)
        result = self.conv3(result)
        result = self.pool2(result)
        result = self.dropout1(result)
        result = self.flatten1(result)
        result = self.dense1(result)
        result = self.dropout2(result)
        result = self.dense2(result)
        return result

    def get_dataset(self, type, batch_size=32):

        def cast_mnist(dictionary):
            dictionary['image'] = tf.cast(dictionary['image'], tf.float32)
            return dictionary

        dataset = tfds.load("mnist", split=type)  # type: tf.data.Dataset
        dataset = dataset.map(cast_mnist)
        dataset = dataset.shuffle(1024).batch(batch_size)
        return dataset

    def test(self, test_data=None):
        """
        :type test_data: tf.data.Dataset
        """
        test_data = self.test_dataset if test_data is None else test_data
        accuracy = tf.contrib.eager.metrics.Accuracy()
        for input in test_data:
            images, labels = input['image'], input['label']
            logits = self(images)
            prediction = tf.math.argmax(logits, axis=1)
            accuracy(labels, prediction)
        print("Evaluation set accuracy is: {:.2%}".format(accuracy.result()))

    def train(self, train_data=None, optimizer=tf.train.AdamOptimizer()):
        """
        :type optimizer: tf.train.Optimizer
        :type train_data: tf.data.Dataset
        """
        train_data = self.train_dataset if train_data is None else train_data
        for batch, input in enumerate(train_data):
            if batch % 100 == 0:
                print(batch)
            images, labels = input['image'], input['label']
            with tf.GradientTape() as tape:
                logits = self(images)
                loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)
            self.loss_history.append(loss_value.numpy())
            grads = tape.gradient(loss_value, self.variables)
            optimizer.apply_gradients(zip(grads, self.variables), global_step=tf.train.get_or_create_global_step())

        plt.plot(self.loss_history)
        plt.show()
        print("Loss value is: {:.2%}".format(self.loss_history[-1]))
        self.test()
        self.save_model_data()

if __name__ == "__main__":
    tf.enable_eager_execution()
    model = MNISTModel()
    model.train()
