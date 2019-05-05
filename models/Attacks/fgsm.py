from tensorflow.python import enable_eager_execution
from tensorflow.python.ops.losses.losses_impl import sparse_softmax_cross_entropy
from tensorflow_datasets import Split
from models.MNISTModels.DenseModel import DenseModel
from models.utils.images import show_plot

import tensorflow as tf

# TODO: Generalize for all models
# jedno krokowe : https://arxiv.org/pdf/1412.6572.pdf
# iteracyjne : dodać żródło ?


def _fgsm(data_sample, model, i, eps, y_target=None, min=0.0, max=1.0):
    image, label = data_sample
    show_plot(model(image), image, model.get_label_names())
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
    show_plot(model(image), image, model.get_label_names())

def untargeted_fgsm(data_sample, model, i, eps, min=0.0, max=1.0):
    return _fgsm(data_sample, model, i, eps, min=min, max=max)

def targeted_fgsm(data_sample, model, i, eps, y_target, min=0.0, max=1.0):
    return _fgsm(data_sample, model, i, eps, y_target=y_target, min=min, max=max)


def test_fgsm_mnist():
    model = DenseModel()
    model.load_model_data()
    eval_dataset = model.get_dataset(Split.TEST, batch_size=1)
    target_label = tf.constant(7, dtype=tf.int64, shape=(1))
    for data_sample in eval_dataset.take(1):
        # targeted_fgsm(data_sample, model, 100, 0.01, target_label)
        untargeted_fgsm(data_sample, model, 1000, 0.0001)

if __name__ == "__main__":
    enable_eager_execution()
    for i in range(3):
        test_fgsm_mnist()
