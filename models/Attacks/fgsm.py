from tensorflow.python import enable_eager_execution, ConfigProto
from tensorflow.python.ops.losses.losses_impl import softmax_cross_entropy
from tensorflow_datasets import Split

from models.Attacks.attack import Attack
from models.CIFAR10Models.ConvModel import ConvModel
from models.ImageNet.InceptionV3Wrapper import ResNetWrapper
from models.MNISTModels.DenseModel import DenseModel
from models.utils.images import show_plot

import tensorflow as tf


@tf.function
def _fgsm(data_sample, model, i_max=1, eps=0.35, target_label=None, min=0.0, max=1.0):
    image, label = data_sample
    eps = eps if target_label is None else -eps
    label = label if target_label is None else target_label
    for i in tf.range(i_max):
        with tf.GradientTape() as tape:
            tape.watch(image)
            logits = model(image)
            loss = softmax_cross_entropy(label, logits)
        gradient = tape.gradient(loss, image)
        image = image + eps * tf.sign(gradient)
        image = tf.clip_by_value(image, min, max)
    return image

def count(model, ret_image, labels):
    ret_labels = model(ret_image)
    ret_labels = tf.one_hot(tf.argmax(ret_labels, axis=1), model.get_number_of_classes())
    has_succeeded = tf.math.reduce_all(tf.math.equal(ret_labels, labels), axis=1)
    has_succeeded = list(has_succeeded.numpy())
    return has_succeeded.count(True)/len(has_succeeded)

def count_untargeted(model, ret_image, labels):
    ret_labels = model(ret_image)
    ret_labels = tf.one_hot(tf.argmax(ret_labels, axis=1), model.get_number_of_classes())
    has_succeeded = tf.math.reduce_all(tf.math.equal(ret_labels, labels), axis=1)
    has_succeeded = list(has_succeeded.numpy())
    return has_succeeded.count(False)/len(has_succeeded)

class FGSM(Attack):

    @staticmethod
    def run_attack(model, data_sample, target_class):
        pass


def test_fgsm_mnist():
    # model = DenseModel().load_model_data()
    model = ConvModel().load_model_data()
    # model = ResNetWrapper()
    score = []
    batch_size = 100
    eval_dataset = model.get_dataset(Split.TEST, batch_size=batch_size, shuffle=1)
    target_label = tf.one_hot(tf.constant([7]*batch_size, dtype=tf.int64), model.get_number_of_classes())

    for data_sample in eval_dataset.take(10):

        ret_image = _fgsm(data_sample, model, i_max=1000, eps=0.0002, target_label=target_label)
        # ret_image = _fgsm(data_sample, model, i_max=1, eps=0.02)

        # print(count_untargeted(model, ret_image, data_sample[1]))
        # score.append(count_untargeted(model, ret_image, data_sample[1]))
        print(count(model, ret_image, target_label))
        score.append(count(model, ret_image, target_label))
        for i in range(1):
            show_plot(model(tf.expand_dims(data_sample[0][i], axis=0)), tf.expand_dims(data_sample[0][i], axis=0), model.get_label_names())
            show_plot(model(tf.expand_dims(ret_image[i], axis=0)), tf.expand_dims(ret_image[i], axis=0), model.get_label_names())

    score_avg = sum(score)/len(score)
    print("final score is: " + str(score_avg))

if __name__ == "__main__":
    test_fgsm_mnist()
