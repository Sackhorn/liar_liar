import errno
import json
import os
from os import path
from random import randrange

import tensorflow as tf
from tensorflow_datasets import Split

from liar_liar.models.base_models.sequential_model import SequentialModel, get_all_models
from liar_liar.utils.attack_metrics import AttackMetricsAccumulator, L2_Metrics, Accuracy, Robustness, \
    Timing
from liar_liar.utils.general_names import *
from liar_liar.utils.generate_side_by_side import show_plot_comparison
from liar_liar.utils.utils import disable_tensorflow_logging, get_results_for_model_and_parameter


def attack_with_params_dict(attack_params, attack_wrapper, targeted, show_plot=False):
    disable_tensorflow_logging()
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
            if len(results_dict) > 0 and get_results_for_model_and_parameter(results_dict, parameter_dict, model.MODEL_NAME) is not None:
                print(("Found results for Model: {} with parameters {} skipping".format(model.MODEL_NAME, json.dumps(parameter_dict))))
                continue
            nmb_classes = model.get_number_of_classes()
            # TODO: Find a way to choose target class this way each parameter set has the same target
            # TODO: also each batch shouldn't have the same class but i don't know if this is acheivable without imapring performance
            target_class = tf.one_hot(randrange(0, nmb_classes), nmb_classes)
            target_class = target_class if targeted else None
            print("MODEL: {} ATTACK: {} PARAMS: {}".format(model.MODEL_NAME, attack_wrapper.__name__, parameter_dict))
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

def try_get_results_dict(attack_wrapper_name):
    if "sanity_check" in attack_wrapper_name:
        return {}
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
    # TODO: this should take into account min and max of model
    dataset = classifier.get_dataset(Split.TEST, shuffle=1, batch_size=batch_size, filter=filter_fnc)
    metrics_accumulator = AttackMetricsAccumulator([Accuracy(), L2_Metrics(), Robustness(), Timing()])
    for data_sample in dataset.take(nmb_elements):
        image, labels = data_sample
        if target_class is not None:
            ret_image, logits, parameters = attack(classifier, data_sample, target_class)
        else:
            ret_image, logits, parameters = attack(classifier, data_sample)

        metrics_dict = metrics_accumulator.accumulate_metrics(image,
                                                              labels,
                                                              ret_image,
                                                              logits,
                                                              batch_size,
                                                              target_class)
        if show_plots:
            show_plot_comparison(adv_image=ret_image[0],
                                 adv_logits=logits[0],
                                 orig_image=image[0],
                                 orig_logits=classifier(tf.expand_dims(image[0], 0)),
                                 labels_names=classifier.get_label_names(),
                                 plot_title=attack.__name__ + " " + classifier.MODEL_NAME,
                                 target_class = target_class,
                                 true_class=labels[0])

        log_string = "TIME_BATCH: {:2f} TIME_PER_SAMPLE {:2f} ATACK_ACC: {:2f} MEAN_L2 {:2f} ROBUSTNESS: {:2f}"
        log_string = log_string.format(float(metrics_dict[TIME_PER_BATCH_KEY]),
                                       float(metrics_dict[AVG_TIME_SAMPLE_KEY]),
                                       float(metrics_dict[ACCURACY_KEY]),
                                       float(metrics_dict[L2_AVERAGE_KEY]),
                                       float(metrics_dict[ROBUSTNESS_KEY]))
        print(log_string)


    results = {
        ACCURACY_KEY : float(metrics_dict[ACCURACY_KEY]),
        L2_MEDIAN_KEY : float(metrics_dict[L2_MEDIAN_KEY]),
        L2_AVERAGE_KEY : float(metrics_dict[L2_AVERAGE_KEY]),
        ROBUSTNESS_KEY : float(metrics_dict[ROBUSTNESS_KEY]),
        PARAMETERS_KEY : parameters,
        AVG_TIME_SAMPLE_KEY : float(float(metrics_dict[AVG_TIME_SAMPLE_KEY])),
        MODEL_NAME_KEY : classifier.MODEL_NAME,
        ATTACK_NAME_KEY : attack.__name__,
    }
    return results