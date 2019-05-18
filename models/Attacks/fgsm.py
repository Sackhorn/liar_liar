from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.ops.losses.losses_impl import sparse_softmax_cross_entropy
from tensorflow_datasets import Split

from models.CIFAR10Models.ConvModel import ConvModel
from models.MNISTModels.DenseModel import DenseModel
from models.utils.images import show_plot

import tensorflow as tf

# TODO: Generalize for all models
# jedno krokowe : https://arxiv.org/pdf/1412.6572.pdf
# iteracyjne : dodać żródło ?


def _fgsm(data_sample, model, i, eps, target_label=None, min=0.0, max=1.0):
    image, label = data_sample
    show_plot(model(image), image, model.get_label_names())
    eps = eps if target_label is None else -eps
    label = label if target_label is None else target_label
    for i in range(i):
        with tf.GradientTape() as tape:
            tape.watch(image)
            logits = model(image)
            loss = categorical_crossentropy(label, logits)
        gradient = tape.gradient(loss, image)
        image = image + eps * tf.sign(gradient)
        image = tf.clip_by_value(image, min, max)
    show_plot(model(image), image, model.get_label_names())

def untargeted_fgsm(data_sample, model, i, eps, min=0.0, max=1.0):
    return _fgsm(data_sample, model, i, eps, min=min, max=max)

def targeted_fgsm(data_sample, model, i, eps, y_target, min=0.0, max=1.0):
    return _fgsm(data_sample, model, i, eps, target_label=y_target, min=min, max=max)


def test_fgsm_mnist():
    model = DenseModel()
    model.load_model_data()
    eval_dataset = model.get_dataset(Split.TEST, batch_size=1)
    target_label = tf.one_hot(tf.constant(7, dtype=tf.int64, shape=(1)), model.get_number_of_classes())
    for data_sample in eval_dataset.take(1):
        # targeted_fgsm(data_sample, model, 100, 0.001, target_label)
        untargeted_fgsm(data_sample, model, 1, 0.05)

if __name__ == "__main__":
    enable_eager_execution()
    for i in range(3):
        test_fgsm_mnist()
