import json
from math import ceil

import numpy as np
import matplotlib.pyplot as plt

from liar_liar.attacks_tests.test_fgsm import params_list as fgsm_params
from liar_liar.attacks_tests.test_deepfool import params_list as deepfool_params
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
    GENERATION_NUMBER : "generation_nmb",
    POPULATION_NMB : "population_nmb",
    MUTATION_PROBABILITY : "mutation_probability",
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

PARAMS_SETS_IN_TABLE = fgsm_params
PRINTABLE_METRICS = ["accuracy_result", "L2_median", "L2_average", "average_time_per_sample"]

def filter_for_params(dict, acceptable_params_in_table):
    new_dict = {}
    for acceptable_params in  acceptable_params_in_table:
        for model, results_list in dict.items():
            for results_set in results_list:
                if results_set["parameters"]["iter_max"] == acceptable_params["iter_max"] and results_set["parameters"]["eps"] == acceptable_params["eps"]:
                    try:
                        new_dict[model].append(results_set)
                    except KeyError:
                        new_dict[model] = []
                        new_dict[model].append(results_set)
    return new_dict

def import_and_print(file_name, nmb_columns_for_params):
    results = None
    with open('' + file_name, 'r') as file:
        results = json.load(file)
    # results = filter_for_params(results, PARAMS_SETS_IN_TABLE)

    #we cell for each metric in each param set
    nmb_metrics = len(PRINTABLE_METRICS)
    nmb_params_sets = len(PARAMS_SETS_IN_TABLE)
    nmb_columns = nmb_columns_for_params * nmb_metrics
    row_columns = r'\begin{tabular}{|c||' + (r'c|' * nmb_columns) + r'}'
    print(row_columns)
    print(r'\hline')

    for fold in range(ceil(nmb_params_sets / nmb_columns_for_params)):
        # a cell for each parameter set
        header_row = ""
        starting_param_index = fold*nmb_columns_for_params
        ending_param_index = (fold+1)*nmb_columns_for_params
        number_of_params_in_fold = len(PARAMS_SETS_IN_TABLE[starting_param_index : ending_param_index])
        for i in range(starting_param_index, starting_param_index + number_of_params_in_fold):
            params_string = ""
            for key, val in PARAMS_SETS_IN_TABLE[i].items():
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
            ending_param_index_overflow = len(PARAMS_SETS_IN_TABLE) if ending_param_index >( len(PARAMS_SETS_IN_TABLE)-1) else ending_param_index
            row = str(MODEL_NAME_MAP[model_name]) + " & "
            for params_idx in range(starting_param_index, ending_param_index_overflow):


                cur_param_set = PARAMS_SETS_IN_TABLE[params_idx]
                results_for_model_and_param = get_results_for_model_and_parameter(results, cur_param_set, model_name)
                for metric in PRINTABLE_METRICS:
                    if metric!="accuracy_result":
                        row += ltx_frmt(results_for_model_and_param[metric]) + " & "
                    else:
                        row += ltx_prcnt(results_for_model_and_param[metric]) + " & "
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
    plt.savefig("fgsm_heat_map.pdf")

    print(r"\begin{left}")
    print(r"\includegraphics[width=0.5\textwidth]{fgsm_heat_map.pdf}")
    print(r"\end{left}")

if __name__ == "__main__":
    PARAMS_SETS_IN_TABLE = deepfool_params
    import_and_print('../../json_results/deepfool_wrapper.json', 2)


