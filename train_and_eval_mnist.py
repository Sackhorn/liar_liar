import tensorflow as tf
import tensorflow_datasets as tfds

from MNISTModel import MNISTModel

tf.enable_eager_execution()

def cast_mnist(dictionary):
    dictionary['image'] = tf.cast(dictionary['image'], tf.float32)
    return dictionary

mnist_train, info = tfds.load("mnist", split=tfds.Split.TRAIN, with_info=True)  # type: (tf.data.Dataset, tfds.core.DatasetInfo)
mnist_train = mnist_train.map(cast_mnist)
mnist_train = mnist_train.shuffle(1024).batch(32)

mnist_eval = tfds.load("mnist", split=tfds.Split.TEST)
mnist_eval = mnist_eval.map(cast_mnist)
mnist_eval = mnist_eval.shuffle(1024).batch(32)

model = MNISTModel()
optimizer = tf.train.AdamOptimizer()
model.train(mnist_train, optimizer)
model.evaluate(mnist_eval)
model.save_weights('/home/sackhorn/tensorflow_models/mnist')

