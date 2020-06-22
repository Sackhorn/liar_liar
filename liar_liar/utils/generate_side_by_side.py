from random import randrange

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_datasets import Split

from liar_liar.attacks.bfgs import bfgs_wrapper
from liar_liar.attacks.c_and_w import carlini_wagner_wrapper
from liar_liar.attacks.deepfool import deepfool_wrapper
from liar_liar.attacks.gen_attack import gen_attack_wrapper
from liar_liar.attacks.jsma import jsma_targeted_wrapper
from liar_liar.base_models.model_names import *
from liar_liar.base_models.sequential_model import get_all_models, SequentialModel
from liar_liar.utils.general_names import *
from liar_liar.utils.utils import find_or_create_file_path


def generate_side_by_side(attack_wrapper,
                          attack_params,
                          file_name,
                          targeted=None,
                          shuffle=100):
    all_models = get_all_models()
    classifier: SequentialModel
    for classifier in all_models:
        try:
            model_dict = attack_params[classifier.MODEL_NAME]
        except KeyError:
            print("{} model not found in interclass params dict".format(classifier.MODEL_NAME))
            continue
        parameters = model_dict[PARAMETERS_KEY]
        dataset = classifier.get_dataset(Split.TEST, batch_size=1, shuffle=shuffle)
        for data_sample in dataset.take(-11):
            image, label = data_sample
            classification = classifier(image)
            classification = tf.argmax(classification, axis=1)
            if tf.math.not_equal(classification, tf.argmax(label, axis=1)):
                continue
            for parameter_set in parameters:
                attack = attack_wrapper(**parameter_set)
                if not targeted:
                    ret_image, logits, _ = attack(classifier, data_sample)
                else:
                    number_of_classes = classifier.get_number_of_classes()
                    one_hot_target_class = tf.one_hot(randrange(0, number_of_classes), number_of_classes)
                    ret_image, logits, _ = attack(classifier, data_sample, one_hot_target_class)
                show_plot_comparison(ret_image,
                                     logits,
                                     image,
                                     classifier(image),
                                     classifier.get_label_names(),
                                     target_class=tf.expand_dims(one_hot_target_class, axis=0) if targeted else None,
                                     true_class=label,
                                     file_name=file_name + classifier.MODEL_NAME)
                break
            break

# TODO: Fix this breaking for running sanity checks
def show_plot_comparison(adv_image,
                         adv_logits,
                         orig_image,
                         orig_logits,
                         labels_names,
                         plot_title=None,
                         target_class=None,
                         true_class=None,
                         file_name=""):
    """

    :type orig_logits: Tensor
    :type orig_image: Tensor
    """
    file_name = find_or_create_file_path(file_name)
    labels_names = np.arange(orig_logits.numpy().size) if labels_names is None else np.array(labels_names)

    orig_logits = orig_logits.numpy().flatten()
    orig_colors = ['blue'] * len(orig_logits)
    adv_logits = adv_logits.numpy().flatten()
    adv_colors = ['blue']*len(adv_logits)

    true_class = int(tf.argmax(true_class, axis=1).numpy())
    orig_colors[true_class], adv_colors[true_class] = 'green', 'green'

    if target_class is not None:
        target_class = int(tf.argmax(target_class, axis=1).numpy()) if target_class is not None else None
        orig_colors[target_class], adv_colors[target_class] = 'red', 'red'
    orig_zip = list(zip(orig_logits, labels_names, orig_colors))
    adv_zip = list(zip(adv_logits, labels_names, adv_colors))
    #case when we have more than 10 classes in dataset
    # if len(orig_logits) > 10:
    orig_zip = sorted(orig_zip, key=lambda x: x[0], reverse=True)
    adv_zip = sorted(adv_zip, key=lambda x: x[0], reverse=True)
    orig_zip = orig_zip[:10]
    adv_zip = adv_zip[:10]
    orig_logits, orig_labels, orig_colors = list(zip(*orig_zip))
    adv_logits, adv_labels, adv_colors = list(zip(*adv_zip))

    figure = plt.figure(figsize=(6,6))
    if plot_title is not None:
        figure.suptitle(plot_title)

    ax = figure.add_subplot(2, 2, 1)
    ax.imshow(tf.squeeze(orig_image), cmap=plt.get_cmap("gray"))
    ax.axis('off')

    ax = figure.add_subplot(2, 2, 2)
    ax.bar(np.arange(len(orig_logits)), orig_logits, color=orig_colors)
    plt.xticks(np.arange(len(orig_logits)), orig_labels, rotation=90)
    ax.axis('tight')

    ax = figure.add_subplot(2, 2, 3)
    ax.imshow(tf.squeeze(adv_image), cmap=plt.get_cmap("gray"))
    ax.axis('off')

    ax = figure.add_subplot(2, 2, 4)
    ax.bar(np.arange(len(adv_logits)), adv_logits, color=adv_colors)
    plt.xticks(np.arange(len(adv_logits)), adv_labels, rotation=90)
    ax.axis('tight')
    # figure.tight_layout()
    if file_name == "":
        plt.show()
    else:
        plt.savefig(file_name)




def bfgs_generate_side_by_side():
    bfgs_params = [{ITER_MAX: 1000}]

    bfgs_model_params = {
        MNIST_TF_NAME:{PARAMETERS_KEY: bfgs_params},
        SIMPLENET_CIFAR10_NAME:{PARAMETERS_KEY: bfgs_params},
        SIMPLENET_CIFAR100_NAME:{PARAMETERS_KEY: bfgs_params},
        INCEPTION_V3_NAME:{PARAMETERS_KEY: bfgs_params},
    }
    generate_side_by_side(bfgs_wrapper, bfgs_model_params, "../../latex/img/side_by_side_bfgs", targeted=True)

def genattack_generate_side_by_side():
    genattack_params = [{GENERATION_NUMBER:10000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}]

    genattack_model_params = {
        MNIST_TF_NAME: {PARAMETERS_KEY: genattack_params},
        SIMPLENET_CIFAR10_NAME: {PARAMETERS_KEY: genattack_params},
        SIMPLENET_CIFAR100_NAME: {PARAMETERS_KEY: genattack_params},
        INCEPTION_V3_NAME: {PARAMETERS_KEY: genattack_params},
    }
    generate_side_by_side(gen_attack_wrapper, genattack_model_params, "../../../latex/img/side_by_side_genattack", targeted=True, shuffle=10)

def deepfool_generate_side_by_side():
    deepfool_params = [{ITER_MAX:1000}]

    deepfool_model_params = {
        MNIST_TF_NAME: {PARAMETERS_KEY: deepfool_params},
        SIMPLENET_CIFAR10_NAME: {PARAMETERS_KEY: deepfool_params},
        SIMPLENET_CIFAR100_NAME: {PARAMETERS_KEY: deepfool_params},
        INCEPTION_V3_NAME: {PARAMETERS_KEY: deepfool_params},
    }
    generate_side_by_side(deepfool_wrapper, deepfool_model_params, "../../../latex/img/side_by_side_deepfool", targeted=False)

def carliniwagner_generate_side_by_side():
    carlini_params = [{OPTIMIZATION_ITER:1000, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0}]

    carlini_model_params = {
        MNIST_TF_NAME: {PARAMETERS_KEY: carlini_params},
        SIMPLENET_CIFAR10_NAME: {PARAMETERS_KEY: carlini_params},
        SIMPLENET_CIFAR100_NAME: {PARAMETERS_KEY: carlini_params},
        INCEPTION_V3_NAME: {PARAMETERS_KEY: carlini_params},
    }
    generate_side_by_side(carlini_wagner_wrapper, carlini_model_params, "../../../latex/img/side_by_side_carlini", targeted=True)

def jsma_targeted_generate_side_by_side():
    jsma_params = [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}]

    jsma_model_params = {
        # MNIST_TF_NAME : {PARAMETERS_KEY: jsma_params},
        # SIMPLENET_CIFAR10_NAME : {PARAMETERS_KEY: jsma_params},
        # SIMPLENET_CIFAR100_NAME : {PARAMETERS_KEY: jsma_params},
        MOBILENETV2_NAME : {PARAMETERS_KEY: jsma_params},
    }
    generate_side_by_side(jsma_targeted_wrapper, jsma_model_params, "../../../latex/img/side_by_side_jsma_targeted", targeted=True)

if __name__ == "__main__":
    # bfgs_generate_side_by_side()
    # genattack_generate_side_by_side()
    # deepfool_generate_side_by_side()
    # carliniwagner_generate_side_by_side()
    jsma_targeted_generate_side_by_side()