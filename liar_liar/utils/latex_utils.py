import json
import sys
import traceback
from math import ceil

import numpy as np
import matplotlib.pyplot as plt

from liar_liar.attacks_experiments.test_bfgs import bfgs_params
from liar_liar.attacks_experiments.test_carlini_wagner import carlini_wagner_params
from liar_liar.attacks_experiments.test_fgsm import params_list as fgsm_params
from liar_liar.attacks_experiments.test_deepfool import params_list as deepfool_params
from liar_liar.attacks_experiments.test_gen_attack import gen_attack_params
from liar_liar.attacks_experiments.test_jsma import jsma_params
from liar_liar.models.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.utils import get_results_for_model_and_parameter

METRICS_NAME_MAP = {
    ACCURACY_KEY : r"ACC",
    L2_MEDIAN_KEY : r"$\operatorname{median}L_2$",
    L2_AVERAGE_KEY : r"$\overline{L_2}$",
    AVG_TIME_SAMPLE_KEY : r"$T$",
    ROBUSTNESS_KEY : r"$\rho_{adw}$"

}

PARAMETERS_NAME_MAP = {
    ITER_MAX : r"$\textit{i}$",
    EPS : r"\(\epsilon\)",
    OPTIMIZATION_ITER : "$i$",
    BINARY_ITER : "$i_{b}$",
    C_HIGH : "c\_high",
    C_LOW : "c\_low",
    KAPPA : r"\(\kappa\)",
    GENERATION_NUMBER : r"$i$",
    POPULATION_NMB : "N",
    MUTATION_PROBABILITY : r"$\alpha$",
    DELTA : r"\(\delta\)",
    MAX_PERTURBATION : "$\epsilon_\max$",
    THETA : r"\(\theta\)",
    IS_INCREASING : "is\_increasing",
    USE_LOGITS : "use\_logits"
}

MODEL_NAME_MAP = {
CIFAR_10_CONV_NAME : "CIFAR-10",
CIFAR_100_CONV_NAME : "CIFAR-100",
INCEPTION_V3_NAME : "InceptionV3",
RESNET_NAME : "resnet50v2",
MNIST_CONV_NAME : "MNIST Convolutional",
MNIST_DENSE_NAME : "MNIST Dense",
LE_NET_NAME : "Le Net 5",
SIMPLENET_CIFAR10_NAME : "SimpleNet CIFAR-10",
SIMPLENET_CIFAR100_NAME : "SimpleNet CIFAR-100",
MOBILENETV2_NAME : "MobileNetV2",
MNIST_TF_NAME : "MNIST TF Model"
}

def ltx_frmt(float):
    return r"{:04.5f}".format(float)
def ltx_prcnt(float):
    return r"{:4.1f}".format(float * 100) + r"\%"
def ltx_acc(float):
    return r"{:4.1f}".format(float * 100) + r"\%"

PRINTABLE_METRICS = [ACCURACY_KEY, L2_MEDIAN_KEY, L2_AVERAGE_KEY, AVG_TIME_SAMPLE_KEY, ROBUSTNESS_KEY]


ACC_METRICS = ["categorical_accuracy", "top_k_categorical_accuracy"]
ACC_METRICS_NAMES = {"categorical_accuracy": "Top-1", "top_k_categorical_accuracy":"Top-5"}

def generate_accuracy_table(file_name):
    main_string = ""
    with open(file_name, 'r') as file:
        acc_results = json.load(file)

    row_columns = r'\begin{tabular}{|c||' + (r'c|' * len(ACC_METRICS)) + r'}'
    main_string += row_columns+ "\n"
    main_string += r'\hline'+ "\n"

    header_row = r"Model Name & "
    for i, metric in enumerate(ACC_METRICS):
        header_row += ACC_METRICS_NAMES[metric]
        if i < len(ACC_METRICS) - 1:
            header_row += r" & "
    header_row += r'\\'
    main_string += header_row+ "\n"
    main_string += r'\hline'+ "\n"

    for model, metrics_dict in acc_results.items():
        row = MODEL_NAME_MAP[model] + "&"
        for i, metric in enumerate(ACC_METRICS):
            try:
                row += ltx_acc(metrics_dict[metric])
            except KeyError:
                row += r" - "
            if i < len(ACC_METRICS) - 1:
                row += r" & "
        row += r"\\"
        main_string += row+ "\n"
        main_string += r'\hline'+ "\n"
    main_string += r'\end{tabular}'+ "\n"
    return main_string

def import_and_print(file_name, nmb_columns_for_params, renderable_params):
    main_string = ""
    results = None
    with open(file_name, 'r') as file:
        results = json.load(file)
    # results = filter_for_params(results, PARAMS_SETS_IN_TABLE)

    #we cell for each metric in each param set
    nmb_metrics = len(PRINTABLE_METRICS)
    nmb_params_sets = len(renderable_params)
    nmb_columns = nmb_columns_for_params * nmb_metrics
    row_columns = r'\begin{tabular}{|c||' + (r'c|' * nmb_columns) + r'}'
    main_string += row_columns+ "\n"
    main_string += r'\hline'+ "\n"

    for fold in range(ceil(nmb_params_sets / nmb_columns_for_params)):
        # a cell for each parameter set
        header_row = ""
        starting_param_index = fold*nmb_columns_for_params
        ending_param_index = (fold+1)*nmb_columns_for_params
        number_of_params_in_fold = len(renderable_params[starting_param_index: ending_param_index])
        for i in range(starting_param_index, starting_param_index + number_of_params_in_fold):
            params_string = ""
            for key, val in renderable_params[i].items():
                params_string += r" {} = {} ".format(PARAMETERS_NAME_MAP[key], str(val))
            header_row += r"& \multicolumn{" + str(nmb_metrics) + r"}{c|}{" + r"{}}}".format(params_string)
        header_row += r" \\ \hline"
        main_string += header_row+ "\n"

        # Create column names
        header_row = r"Model Name & "
        for i in range(nmb_columns):
            header_row +=(METRICS_NAME_MAP[PRINTABLE_METRICS[i % nmb_metrics]])
            if i < nmb_columns -1:
                header_row += r' & '
        header_row += r'\\ \hline'
        main_string += header_row+ "\n"

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
            main_string += row+ "\n"
    main_string += r'\end{tabular}'+ "\n"
    return main_string

def create_heat_map_fgsm_targeted(file_name):
    results = None
    eps_set = set()
    iter_set = set()
    with open('../../json_results/' + file_name, 'r') as file:
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

def print_on_no_except(func):
    def no_except(*args, **kwargs):
        try:
            main_string = func(*args, **kwargs)
            print(main_string)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            sys.stderr.write(str(e))
            sys.stderr.write(func.__name__ + "failed")
            return

    return no_except

@print_on_no_except
def generate_fgsm_table(path='../json_results/FGSMUntargeted.json', nmb_columns=2):
    renderable_params = filter(lambda dict : dict[ITER_MAX] == 1, fgsm_params)
    renderable_params = sorted(renderable_params, key=lambda dict : dict[EPS])
    return import_and_print(path, nmb_columns, renderable_params)

@print_on_no_except
def generate_ifgsm_table(path='../json_results/FGSMUntargeted.json', nmb_columns=2):
    renderable_params = filter(lambda dict: dict[ITER_MAX] > 1, fgsm_params)
    renderable_params = sorted(renderable_params, key=lambda dict: (dict[ITER_MAX], dict[EPS]))
    return import_and_print(path, nmb_columns, renderable_params)

@print_on_no_except
def generate_llfgsm_table(path='../json_results/FGSMTargeted.json', nmb_columns=2):
    renderable_params = sorted(fgsm_params, key=lambda dict: (dict[ITER_MAX], dict[EPS]))
    return import_and_print(path, nmb_columns, renderable_params)

@print_on_no_except
def generate_getattack_table(path='../json_results/GenAttack.json', nmb_columns=1):
    renderable_params = sorted(gen_attack_params, key=lambda dict: dict[GENERATION_NUMBER])
    return import_and_print(path, nmb_columns, renderable_params)

@print_on_no_except
def generate_deepfool_table(path='../json_results/DeepFool.json', nmb_columns=1):
    renderable_params = sorted(deepfool_params, key=lambda dict: dict[ITER_MAX])
    return import_and_print(path, nmb_columns, renderable_params)

@print_on_no_except
def generate_bfgs_table(path='../json_results/BFGS.json', nmb_columns=2):
    renderable_params = sorted(bfgs_params, key=lambda dict: dict[ITER_MAX])
    return import_and_print(path, nmb_columns, renderable_params)

@print_on_no_except
def generate_carlini_table(path='../json_results/CarliniWagner.json', nmb_columns=2):
    renderable_params = sorted(carlini_wagner_params, key=lambda dict: dict[OPTIMIZATION_ITER])
    for param_set in renderable_params:
        del param_set[C_HIGH]
        del param_set[C_LOW]
    return import_and_print(path, nmb_columns, renderable_params)

@print_on_no_except
def generate_jsma_table(path='../json_results/JSMATargeted.json', nmb_columns=1):
    renderable_params = [{MAX_PERTURBATION: 0.1, THETA: 1}]
    return import_and_print(path, nmb_columns, renderable_params)

@print_on_no_except
def generate_accuracy_table_for_all(path = '../json_results/accuracy_results.json'):
    return generate_accuracy_table(path)