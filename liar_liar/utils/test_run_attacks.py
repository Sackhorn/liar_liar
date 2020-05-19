import errno
import json
import os
import time
from random import randrange
from os import path

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.metrics import CategoricalAccuracy
from tensorflow_datasets import Split
from liar_liar.base_models.sequential_model import SequentialModel, get_all_models
from liar_liar.utils.general_names import *
from liar_liar.utils.images import show_plot, show_plot_comparison
from liar_liar.utils.utils import batch_image_norm, disable_logging


def attack_with_params_dict(attack_params, attack_wrapper, targeted, show_plot=False):
    disable_logging()
    results_arr = []
    all_models = get_all_models()
    for model in all_models:
        try:
            model_dict = attack_params[model.MODEL_NAME]
        except KeyError:
            continue
        batches = model_dict[DATASET_KEY]
        parameters = model_dict[PARAMETERS_KEY]
        for parameter_set in parameters:
            nmb_classes = model.get_number_of_classes()
            target_class = tf.one_hot(randrange(0, nmb_classes), nmb_classes) #TODO: Find a way to choose target class
            target_class = target_class if targeted else None
            results = run_test(model,
                     attack_wrapper(**parameter_set),
                     target_class=target_class,
                     batch_size=batches[BATCHES_KEY],
                     nmb_elements=batches[NMB_ELEMENTS_KEY],
                     show_plots=show_plot)
            results_arr.append(results)
        #We do this after every completed test to store information in case of fufure error
        create_results_json(attack_wrapper.__name__, results_arr)

def create_results_json(attack_wrapper_name, results_arr):
    json_file_path = path.dirname(path.realpath(__file__))
    json_file_path = path.join(json_file_path, path.pardir, path.pardir, "json_results", attack_wrapper_name + ".json")
    if not path.exists(path.dirname(json_file_path)):
        try:
            os.mkdir(path.dirname(json_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(results_arr, f, ensure_ascii=False, indent=4)


def run_test(classifier, attack, batch_size, target_class, nmb_elements=None, show_plots=True):
    """

    Args:
        classifier (SequentialModel): A classifier we attack
        attack: A wrapped attack method
        targeted: wether the attack is targeted or not
        batch_size: number of elements being put at once into the attack
        nmb_elements: number of batches we want to run through attack
    """
    # Filter to remove all examples that are misclassified by classifier
    def remove_misclassified_and_target_class(image, label):
        image = tf.expand_dims(image, 0)
        classification = tf.one_hot(tf.argmax(classifier(image), 1), classifier.get_number_of_classes())
        classified_fine = tf.math.reduce_all(tf.math.equal(classification, label))
        in_target_class = tf.math.reduce_all(tf.math.equal(target_class, label))
        ret_val = tf.math.logical_and(classified_fine, tf.logical_not(in_target_class))
        return ret_val

    # Filter to remove just misclassified examples
    def remove_misclassified(image, label):
        image = tf.expand_dims(image, 0)
        classification = tf.one_hot(tf.argmax(classifier(image), 1), classifier.get_number_of_classes())
        return tf.math.reduce_all(tf.math.equal(classification, label))

    filter_fnc = remove_misclassified if target_class is None else remove_misclassified_and_target_class
    accuracy = CategoricalAccuracy()
    l2_distance = np.array([])
    mean_time_per_sample = np.array([])
    dataset = classifier.get_dataset(Split.TEST, shuffle=1, batch_size=batch_size, filter=filter_fnc)
    print("MODEL: {} ATTACK: {}".format(classifier.MODEL_NAME, attack.__name__))
    for data_sample in dataset.take(nmb_elements):
        image, labels = data_sample
        start = time.time()
        if target_class is not None:
            ret_image, logits, parameters = attack(classifier, data_sample, target_class)
            accuracy.update_state(target_class, logits)
            accuracy_result = accuracy.result().numpy()
        else:
            ret_image, logits, parameters = attack(classifier, data_sample)
            accuracy.update_state(labels, logits) #TODO: This is wrong for deepfool with imagenet
            accuracy_result = 1.0 - accuracy.result().numpy()

        l2_distance = np.append(l2_distance, batch_image_norm(image - ret_image).numpy().flatten())
        cur_l2_median = np.median(l2_distance)
        cur_l2_average = np.mean(l2_distance)

        if show_plots:
            show_plot_comparison(adv_image=ret_image[0],
                                 adv_logits=logits[0],
                                 orig_image=image[0],
                                 orig_logits=classifier(tf.expand_dims(image[0], 0)),
                                 labels_names=classifier.get_label_names(),
                                 plot_title=attack.__name__ + " " + classifier.MODEL_NAME,
                                 target_class = target_class,
                                 true_class=labels[0])

        cur_time_per_batch = time.time() - start
        cur_time_per_sample = cur_time_per_batch / batch_size
        mean_time_per_sample = np.append(mean_time_per_sample, cur_time_per_sample)

        print("TIME_BATCH: {:2f} TIME_PER_SAMPLE {:2f} ATACK_ACC: {:2f} MEDIAN_L2 {:2f} MEAN_L2 {:2f}"
              .format(cur_time_per_batch, cur_time_per_sample, accuracy_result, float(cur_l2_median), float(cur_l2_average)))

    results = {
        "accuracy_result": float(accuracy_result),
        "L2_median": float(cur_l2_median),
        "L2_average": float(cur_l2_average),
        "parameters": parameters,
        "average_time_per_batch": float(np.mean(mean_time_per_sample)),
        "model_name": classifier.MODEL_NAME,
        "attack_name": attack.__name__
    }
    return results