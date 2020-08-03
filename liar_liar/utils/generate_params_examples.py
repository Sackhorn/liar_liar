from matplotlib.figure import Figure
from tensorflow_datasets import Split

from liar_liar.attacks.fgsm import fgsm_untargeted_wrapper
from liar_liar.models.base_models.sequential_model import SequentialModel, get_all_models
from liar_liar.utils.general_names import *

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def get_same_image_with_different_params(attack_wrapper,
                                         attack_params,
                                         file_name,
                                         target_class=None,
                                         x_labels=None):
    all_models = get_all_models()
    classifier: SequentialModel
    for classifier in all_models:
        try:
            model_dict = attack_params[classifier.MODEL_NAME]
        except KeyError:
            print("{} model not found in interclass params dict".format(classifier.MODEL_NAME))
            continue
        parameters = model_dict[PARAMETERS_KEY]
        dataset = classifier.get_dataset(Split.TEST, batch_size=1, shuffle=100)
        images_list = []
        for data_sample in dataset.take(-1):
            image, label = data_sample
            label = tf.argmax(label, axis=1)
            classification = classifier(image)
            classification = tf.argmax(classification, axis=1)
            if tf.math.not_equal(classification, label):
                continue
            for parameter_set in parameters:
                attack = attack_wrapper(**parameter_set)
                if target_class is None:
                    ret_image, logits, _ = attack(classifier, data_sample)
                else:
                    ret_image, logits, _ = attack(classifier, data_sample, target_class)
                argmax_logits = tf.argmax(logits, axis=1)
                images_list.append((ret_image, logits))
            images_list.append((data_sample[0], classifier(data_sample[0])))
            break
        plot_multiple_images_in_row(images_list,
                                    classifier.get_label_names(),
                                    file_name + classifier.MODEL_NAME,
                                    x_labels)

def plot_multiple_images_in_row(images_list, labels, file_name, x_labels=None):
    figure: Figure = plt.figure(figsize=(20, 6.5))
    length = len(images_list)
    for i, data_sample in enumerate(images_list):
        image, logits = data_sample
        logits = logits.numpy().flatten()
        logits_labels = list(zip(logits, labels))
        logits_labels = sorted(logits_labels, key=lambda x: x[0], reverse=True)
        logits_labels = logits_labels[:10]
        logits, sorted_labels = list(zip(*logits_labels))

        plt.subplot(2, length, i+1)
        plt.imshow(tf.squeeze(image), cmap=plt.get_cmap("gray"))
        plt.axis('off')

        plt.subplot(2, length, length + i + 1)
        plt.bar(np.arange(len(logits)), logits)
        plt.xticks(np.arange(len(logits)), sorted_labels, rotation=90)
        plt.axis('tight')
        if x_labels is not None:
            plt.xlabel(x_labels[i])

    figure.tight_layout()
    figure.savefig(file_name)


fgsm_params_row_images = [
    {ITER_MAX: 1, EPS: 1.0},
    {ITER_MAX: 1, EPS: 0.1},
    {ITER_MAX: 1, EPS: 0.01},
    {ITER_MAX: 1, EPS: 0.001},
    {ITER_MAX: 1, EPS: 0.0001}
]

fgsm_params_row = {
MNIST_CONV_NAME:
    {
        PARAMETERS_KEY: fgsm_params_row_images
    },
CIFAR_10_CONV_NAME:
    {
        PARAMETERS_KEY: fgsm_params_row_images
    },
CIFAR_100_CONV_NAME:
    {
        PARAMETERS_KEY: fgsm_params_row_images
    },
INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY: fgsm_params_row_images
    },
}

def generate_fgsm_row_images():
    x_labels = list(map(
        lambda x: "{:f}".format(x[EPS]).rstrip("0"),
        fgsm_params_row_images))
    x_labels.append("0")
    x_labels = list(map(lambda x: r"$\epsilon = {}$".format(x), x_labels))
    get_same_image_with_different_params(fgsm_untargeted_wrapper,
                                         fgsm_params_row,
                                         "../../latex/img/row_fgsm_",
                                         x_labels=x_labels)

iteravitve_fgsm_params_row_images = [
    {ITER_MAX: 10, EPS: 0.001},
    {ITER_MAX: 100, EPS: 0.001},
    {ITER_MAX: 1000, EPS: 0.00001},
    {ITER_MAX: 1000, EPS: 0.0001}
]

iterative_fgsm_params_row = {
MNIST_CONV_NAME:
    {
        PARAMETERS_KEY: iteravitve_fgsm_params_row_images
    },
CIFAR_10_CONV_NAME:
    {
        PARAMETERS_KEY: iteravitve_fgsm_params_row_images
    },
CIFAR_100_CONV_NAME:
    {
        PARAMETERS_KEY: iteravitve_fgsm_params_row_images
    },
INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY: iteravitve_fgsm_params_row_images
    },
}

def generate_iterative_fgsm_row_images():
    x_labels = list(map(
        lambda x: ("{:f}".format(x[EPS]).rstrip("0"), x[ITER_MAX]),
        iteravitve_fgsm_params_row_images))
    x_labels = list(map(lambda epsi: r"$\epsilon = {},i = {}$".format(epsi[0], epsi[1]), x_labels))
    x_labels.append("oryginał")
    get_same_image_with_different_params(fgsm_untargeted_wrapper,
                                         iterative_fgsm_params_row,
                                         "../../latex/img/row_iterative_fgsm_",
                                         x_labels=x_labels)

if __name__ == "__main__":
    generate_fgsm_row_images()
    generate_iterative_fgsm_row_images()