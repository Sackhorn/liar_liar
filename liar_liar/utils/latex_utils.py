import json
import os
from math import ceil

import numpy as np
import matplotlib.pyplot as plt

from liar_liar.attacks.fgsm import fgsm_targeted_wrapper
from liar_liar.attacks_tests.test_fgsm import params_list as fgsm_params
from liar_liar.attacks_tests.test_deepfool import params_list as deepfool_params
from liar_liar.attacks_tests.test_gen_attack import gen_attack_params
from liar_liar.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.utils import get_results_for_model_and_parameter

METRICS_NAME_MAP = {
    "accuracy_result" : "Accuracy",
    "L2_median" : "L2 Median",
    "L2_average" : "L2 Average",
    "average_time_per_sample" : "Time per sample",
}

PARAMETERS_NAME_MAP = {
    ITER_MAX : r"i",
    EPS : r"\(\epsilon\)",
    OPTIMIZATION_ITER : "optimization_iter",
    BINARY_ITER : "binary_iter",
    C_HIGH : "c_high",
    C_LOW : "c_low",
    KAPPA : "kappa",
    GENERATION_NUMBER : "grtn nmbr",
    POPULATION_NMB : "pop nmbr",
    MUTATION_PROBABILITY : "mutn prob",
    DELTA : "delta",
    MAX_PERTURBATION : "max_perturbation",
    THETA : "theta",
    IS_INCREASING : "is_increasing",
    USE_LOGITS : "use_logits"
}

MODEL_NAME_MAP = {
CIFAR_10_CONV_NAME : "CIFAR-10",
CIFAR_100_CONV_NAME : "CIFAR-100",
INCEPTION_V3_NAME : "ImagenetV3",
RESNET_NAME : "resnet50v2",
MNIST_CONV_NAME : "MNIST Convolutional",
MNIST_DENSE_NAME : "MNIST Dense",
LE_NET_NAME : "Le Net 5"
}

def ltx_frmt(float):
    return r"{:04.5f}".format(float)
def ltx_prcnt(float):
    return r"{:04.5f}".format(float * 100) + r"\%"

PRINTABLE_METRICS = ["accuracy_result", "L2_median", "L2_average", "average_time_per_sample"]

def import_and_print(file_name, nmb_columns_for_params, renderable_params):
    results = None
    with open(file_name, 'r') as file:
        results = json.load(file)
    # results = filter_for_params(results, PARAMS_SETS_IN_TABLE)

    #we cell for each metric in each param set
    nmb_metrics = len(PRINTABLE_METRICS)
    nmb_params_sets = len(renderable_params)
    nmb_columns = nmb_columns_for_params * nmb_metrics
    row_columns = r'\begin{tabular}{|c||' + (r'c|' * nmb_columns) + r'}'
    print(row_columns)
    print(r'\hline')

    for fold in range(ceil(nmb_params_sets / nmb_columns_for_params)):
        # a cell for each parameter set
        header_row = ""
        starting_param_index = fold*nmb_columns_for_params
        ending_param_index = (fold+1)*nmb_columns_for_params
        number_of_params_in_fold = len(renderable_params[starting_param_index: ending_param_index])
        for i in range(starting_param_index, starting_param_index + number_of_params_in_fold):
            params_string = ""
            for key, val in renderable_params[i].items():
                params_string += r" {}={} ".format(PARAMETERS_NAME_MAP[key], str(val))
            header_row += r"& \multicolumn{" + str(nmb_metrics) + r"}{c|}{" + r"{}}}".format(params_string)
        header_row += r" \\ \hline"
        print(header_row)

        # Create column names
        header_row = r"Model Name & "
        for i in range(nmb_columns):
            header_row +=(METRICS_NAME_MAP[PRINTABLE_METRICS[i % nmb_metrics]])
            if i < nmb_columns -1:
                header_row += r' & '
        header_row += r'\\ \hline'
        print(header_row)

        #Fill rows with data

        for model_name, results_list in results.items():
            ending_param_index_overflow = len(renderable_params) if ending_param_index > (len(renderable_params) - 1) else ending_param_index
            row = str(MODEL_NAME_MAP[model_name]) + " & "
            for params_idx in range(starting_param_index, ending_param_index_overflow):


                cur_param_set = renderable_params[params_idx]
                results_for_model_and_param = get_results_for_model_and_parameter(results, cur_param_set, model_name)
                for metric in PRINTABLE_METRICS:
                    try:
                        metric_result = results_for_model_and_param[metric]
                    except TypeError:
                        metric_result = -1.0
                    if metric!="accuracy_result":
                        row += ltx_frmt(metric_result) + " & "
                    else:
                        row += ltx_prcnt(metric_result) + " & "
            row = row[:-3] #remove last ampersand
            row += r"\\ \hline"
            print(row)
    print(r'\end{tabular}')

def create_heat_map_fgsm_targeted(file_name):
    results = None
    eps_set = set()
    iter_set = set()
    with open('../json_results/' + file_name, 'r') as file:
        results = json.load(file)
    params_list_cifar = results[CIFAR_10_CONV_NAME]
    for result in params_list_cifar:
        iter_set.add(result["parameters"]["iter_max"])
        eps_set.add(result["parameters"]["eps"])
    eps_set = sorted(eps_set)
    iter_set = sorted(iter_set)
    acc_heat_map = []
    l2_heat_map = []
    for iter in iter_set:
        acc_submap = []
        l2_submap = []
        for eps in eps_set:
            for dict in params_list_cifar:
                if dict["parameters"]["eps"] == eps and dict["parameters"]["iter_max"] == iter:
                    acc_submap.append(dict["accuracy_result"])
                    l2_submap.append(dict["L2_average"])
        acc_heat_map.append(acc_submap)
        l2_heat_map.append(l2_submap)
    acc_heat_map = np.array(acc_heat_map)
    l2_heat_map = np.array(l2_heat_map)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
    im = ax1.imshow(acc_heat_map)
    im1 = ax2.imshow(l2_heat_map)

    # We want to show all ticks...
    ax1.set_yticks(np.arange(len(iter_set)))
    ax1.set_xticks(np.arange(len(eps_set)))

    ax2.set_yticks(np.arange(len(iter_set)))
    ax2.set_xticks(np.arange(len(eps_set)))

    # ... and label them with the respective list entries
    ax1.set_yticklabels(list(map(lambda x: str(x), iter_set)))
    ax1.set_xticklabels(list(map(lambda x: str(x), eps_set)))

    ax2.set_yticklabels(list(map(lambda x: str(x), iter_set)))
    ax2.set_xticklabels(list(map(lambda x: str(x), eps_set)))
    for i in range(len(iter_set)):
        for j in range(len(eps_set)):
            text = ax1.text(j, i, "{:04.2f}".format(float(acc_heat_map[i, j])),
                           ha="center", va="center", color="red")
            text = ax2.text(j, i, "{:04.2f}".format(float(l2_heat_map[i, j])),
                            ha="center", va="center", color="red")
    fig.tight_layout()
    plt.savefig("img/fgsm_heat_map.png")

    print(r"\begin{left}")
    print(r"\includegraphics[width=0.5\textwidth]{img/fgsm_heat_map.png}")
    print(r"\end{left}")

def generate_fgsm_table(path='../json_results/fgsm_untargeted_wrapper.json', nmb_columns=2):
    renderable_params = filter(lambda dict : dict[ITER_MAX] == 1, fgsm_params)
    renderable_params = sorted(renderable_params, key=lambda dict : dict[EPS])
    import_and_print(path, nmb_columns, renderable_params)

def generate_ifgsm_table(path='../json_results/fgsm_untargeted_wrapper.json', nmb_columns=3):
    renderable_params = sorted(fgsm_params, key=lambda dict: dict[ITER_MAX])
    import_and_print(path, nmb_columns, renderable_params)

def generate_llfgsm_table(path='../json_results/fgsm_targeted_wrapper.json', nmb_columns=3):
    renderable_params = sorted(fgsm_params, key=lambda dict: dict[ITER_MAX])
    import_and_print(path, nmb_columns, renderable_params)

def generate_getattack_table(path='../json_results/gen_attack_wrapper.json', nmb_columns=2):
    renderable_params = sorted(gen_attack_params, key=lambda dict: dict[GENERATION_NUMBER])
    import_and_print(path, nmb_columns, renderable_params)
