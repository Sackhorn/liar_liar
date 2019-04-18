import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from models.BaseModels.SequentialModel import SequentialModel


class MNISTModel(SequentialModel):

    def __init__(self, MODEL_NAME=""):
        super(MNISTModel, self).__init__(MODEL_NAME=MODEL_NAME)
        self.train_dataset = self.get_dataset(tfds.Split.TRAIN)
        self.test_dataset = self.get_dataset(tfds.Split.TEST)

    def get_dataset(self, type, batch_size=32):
        def cast_mnist(dictionary):
            dictionary['image'] = tf.cast(dictionary['image'], tf.float32)
            dictionary['image'] = tf.div(
                tf.subtract(dictionary['image'], tf.reduce_min(dictionary['image'])),
                tf.subtract(tf.reduce_max(dictionary['image']), tf.reduce_min(dictionary['image'])))
            # dictionary['image'] = tf.subtract(tf.constant(1.0), dictionary['image'])
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
        print("Loss value is: {:.10}".format(self.loss_history[-1]))
        self.test()
        self.save_model_data()
