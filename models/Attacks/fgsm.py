from tensorflow.python import enable_eager_execution, Tensor
from tensorflow.python.ops.losses.losses_impl import sparse_softmax_cross_entropy

from models.MNISTModels.ConvModel import ConvModel
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# TODO: Generalize for all models
# jedno krokowe : https://arxiv.org/pdf/1412.6572.pdf
# iteracyjne : dodać żródło ?



def show_plot(logits, image):
    """

    :type image: Tensor
    """
    probs = tf.nn.softmax(logits)
    probs = probs.numpy().reshape(10).tolist()
    fig = plt.figure()
    img_plt = fig.add_subplot(121)
    # img_plt.imshow(image.numpy().reshape(28, 28).astype(np.float32), cmap=plt.get_cmap("gray"))
    img_plt.imshow(tf.squeeze(image), cmap=plt.get_cmap("gray"))
    bar_plt = fig.add_subplot(122)
    bar_plt.bar(np.arange(10), probs)
    bar_plt.set_xticks(np.arange(10))
    plt.show()

def _fgsm(data_sample, model, i, eps, y_target=None, min=0.0, max=1.0):
    image, label = data_sample
    show_plot(model(image), image)
    eps = eps if y_target is None else -eps
    label = tf.argmax(label, axis=1) if y_target is None else y_target
    for i in range(i):
        with tf.GradientTape() as tape:
            tape.watch(image)
            logits = model(image)
            loss = sparse_softmax_cross_entropy(label, logits)
        gradient = tape.gradient(loss, image)
        image = image + eps * tf.sign(gradient)
        image = tf.clip_by_value(image, min, max)
    show_plot(model(image), image)
    return tf.reduce_all(tf.not_equal(label, tf.math.argmax(logits)))

def untargeted_fgsm(data_sample, model, i, eps, min=0.0, max=1.0):
    return _fgsm(data_sample, model, i, eps, min=min, max=max)

def targeted_fgsm(data_sample, model, i, eps, y_target, min=0.0, max=1.0):
    return _fgsm(data_sample, model, i, eps, y_target=y_target, min=min, max=max)


def test_fgsm_mnist():
    model = ConvModel()
    model.load_model_data()
    eval_dataset = model.get_dataset(tfds.Split.TEST, batch_size=1)
    target_label = tf.constant(7, dtype=tf.int64, shape=(1))
    for data_sample in eval_dataset.take(1):
        # targeted_fgsm(data_sample, model, 100, 0.01, target_label)
        untargeted_fgsm(data_sample, model, 1000, 0.0001)

if __name__ == "__main__":
    enable_eager_execution()
    for i in range(3):
        test_fgsm_mnist()
