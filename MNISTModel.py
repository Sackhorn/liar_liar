import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


NMB_CLASSES = 10
SAVE_DIR = '/home/sackhorn/tensorflow_models/mnist'

class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
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
        self.eval_dataset = self.get_dataset(tfds.Split.TEST)

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

    def evaluate(self, eval_data):
        """

        :type eval_data: tf.data.Dataset
        """
        accuracy = tf.contrib.eager.metrics.Accuracy()
        for input in eval_data:
            images, labels = input['image'],input['label']
            logits = self(images)
            prediction = tf.math.argmax(logits, axis=1)
            accuracy(labels, prediction)
        print("Eval set accuracy is: {:.2%}".format(accuracy.result()))


    def get_dataset(self, type, batch_size=32):
        def cast_mnist(dictionary):
            dictionary['image'] = tf.cast(dictionary['image'], tf.float32) / 255.0
            return dictionary
        dataset, info = tfds.load("mnist", split=type, with_info=True)  # type: (tf.data.Dataset, tfds.core.DatasetInfo)
        dataset = dataset.map(cast_mnist)
        dataset = dataset.shuffle(1024).batch(batch_size)
        return dataset

    def load_model(self):
        self.load_weights(SAVE_DIR)
        self.evaluate(self.eval_dataset)

    def train(self, train_data=None, optimizer=tf.train.AdamOptimizer()):
        """
        :type optimizer: tf.train.Optimizer
        :type train_data: tf.data.Dataset
        """
        train_data = self.train_dataset if train_data is None else train_data
        for batch, input in enumerate(train_data):
            if batch%100==0:
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
        self.save_weights(SAVE_DIR)
