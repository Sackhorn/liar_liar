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
from liar_liar.utils.images import show_plot, show_plot_comparison, show_plot_target_class_grid
from liar_liar.utils.utils import batch_image_norm, disable_logging


def attack_with_params_dict(attack_params, attack_wrapper, targeted, show_plot=False):
    disable_logging()
    results_dict = try_get_results_dict(attack_wrapper.__name__)
    all_models = get_all_models()
    for model in all_models:
        try:
            model_dict = attack_params[model.MODEL_NAME]
        except KeyError:
            continue
        batches = model_dict[DATASET_KEY]
        parameters = model_dict[PARAMETERS_KEY]
        for parameter_dict in parameters:
            if result_exist(results_dict, parameter_dict, model.MODEL_NAME):
                continue
            nmb_classes = model.get_number_of_classes()
            # TODO: Find a way to choose target class this way each parameter set has the same target
            # TODO: also each batch shouldn't have the same class but i don't know if this is acheivable without imapring performance
            target_class = tf.one_hot(randrange(0, nmb_classes), nmb_classes)
            target_class = target_class if targeted else None
            results = run_test(model,
                     attack_wrapper(**parameter_dict),
                     target_class=target_class,
                     batch_size=batches[BATCHES_KEY],
                     nmb_elements=batches[NMB_ELEMENTS_KEY],
                     show_plots=show_plot)
            try:
                results_dict[results[MODEL_NAME_KEY]].append(results)
            except KeyError:
                results_dict[results[MODEL_NAME_KEY]] = [results]
            #We do this after every completed test to store information in case of fufure error
            create_results_json(attack_wrapper.__name__, results_dict)

def result_exist(result_dict, parameter_dict, model_name):
    exists = True
    try:
        results_for_model = result_dict[model_name]
    except KeyError:
        return False

    for per_params_results in results_for_model:
        per_params_results = per_params_results[PARAMETERS_KEY]
        for parameter, val in parameter_dict.items():
            exists &= per_params_results[parameter] == val
        if exists:
            print("Found results for Model: {} with parameters {} skipping".format(model_name, json.dumps(parameter_dict)))
            return True
        exists = True
    return False

def try_get_results_dict(attack_wrapper_name):
    json_file_path = path.dirname(path.realpath(__file__))
    json_file_path = path.join(json_file_path, path.pardir, path.pardir, "json_results", attack_wrapper_name + ".json")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except IOError:
        return {}

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

def interclass_run_with_params_dict(attack_params, attack_wrapper):
    disable_logging()
    all_models = get_all_models()
    classifier: SequentialModel
    for classifier in all_models:
        try:
            model_dict = attack_params[classifier.MODEL_NAME]
        except KeyError:
            continue
        parameters = model_dict[PARAMETERS_KEY]

        for parameter_set in parameters:
            attack = attack_wrapper(**parameter_set)
            nmb_classes = classifier.get_number_of_classes()
            nmb_classes = 10 if nmb_classes >= 10 else nmb_classes
            true_class_dict = {}

            for true_class in range(nmb_classes):
                true_class_dict[true_class] = {}
                true_class_one_hot = tf.one_hot(true_class, classifier.get_number_of_classes())
                def get_baseclass_not_misclassfied(image, label):
                    image = tf.expand_dims(image, 0)
                    classification = tf.one_hot(tf.argmax(classifier(image), 1), classifier.get_number_of_classes())
                    classified_fine = tf.math.reduce_all(tf.math.equal(classification, label))
                    in_true_class = tf.math.reduce_all(tf.math.equal(true_class_one_hot, label))
                    ret_val = tf.math.logical_and(classified_fine, in_true_class)
                    return ret_val

                dataset = classifier.get_dataset(Split.TEST,
                                                 batch_size=1,
                                                 shuffle=1,
                                                 filter=get_baseclass_not_misclassfied)

                range_nmb_target_classes = list(range(nmb_classes))
                range_nmb_target_classes.remove(true_class)
                for target_class in range_nmb_target_classes:
                    target_class_one_hot = tf.one_hot(target_class, classifier.get_number_of_classes())
                    found_sample = False
                    for retry in range(10):
                        for data_sample in dataset.take(1):
                            true_class_dict[true_class][true_class] = data_sample[0]
                            ret_image, logits, _ = attack(classifier, data_sample, target_class_one_hot)
                            if tf.math.reduce_all(tf.math.equal(tf.argmax(logits, 1), tf.argmax(target_class_one_hot))):
                                found_sample = True
                                true_class_dict[true_class][target_class] = ret_image
                        if found_sample:
                            break
                        print("run out of retries returning function without results")
                        return

        show_plot_target_class_grid(true_class_dict)

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
    total_time = time.time()
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
    total_time = time.time() - total_time
    results = {
        ACCURACY_KEY: float(accuracy_result),
        L2_MEDIAN_KEY : float(cur_l2_median),
        L2_AVERAGE_KEY: float(cur_l2_average),
        PARAMETERS_KEY: parameters,
        AVG_TIME_SAMPLE_KEY: float(np.mean(mean_time_per_sample)),
        MODEL_NAME_KEY: classifier.MODEL_NAME,
        ATTACK_NAME_KEY: attack.__name__,
        TOTAL_TIME_KEY: total_time
    }
    return results