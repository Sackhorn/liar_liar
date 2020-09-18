import json
import os.path

import matplotlib.pyplot as plt

from liar_liar.models.base_models.model_names import *
from liar_liar.utils.general_names import *

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
    return r"{:04.2f}".format(float * 100) + r"\%"

PRINTABLE_METRICS = [ACCURACY_KEY, L2_MEDIAN_KEY, L2_AVERAGE_KEY, AVG_TIME_SAMPLE_KEY, ROBUSTNESS_KEY]


ACC_METRICS = ["categorical_accuracy", "top_k_categorical_accuracy"]
ACC_METRICS_NAMES = {"categorical_accuracy": "Top-1", "top_k_categorical_accuracy":"Top-5"}

MNIST_NAME = "mnist"
CIFAR_10_NAME = "cifar-10"
CIFAR_100_NAME = "cifar-100"
IMAGENET_NAME = "imagenet"

MODELS_TO_DATASETS = {
    MNIST_TF_NAME : MNIST_NAME,
    LE_NET_NAME : MNIST_NAME,
    SIMPLENET_CIFAR10_NAME : CIFAR_10_NAME,
    SIMPLENET_CIFAR100_NAME : CIFAR_100_NAME,
    INCEPTION_V3_NAME : IMAGENET_NAME,
    MOBILENETV2_NAME : IMAGENET_NAME,
}

MODELS_TO_CLASSES_NMB = {
    MNIST_TF_NAME : 10,
    LE_NET_NAME : 10,
    SIMPLENET_CIFAR10_NAME : 10,
    SIMPLENET_CIFAR100_NAME : 100,
    INCEPTION_V3_NAME : 1000,
    MOBILENETV2_NAME : 1000,
}

MODELS_TO_NMB_FEATURES = {
    MNIST_TF_NAME: 28*28,
    LE_NET_NAME: 28*28,
    SIMPLENET_CIFAR10_NAME: 32*32*3,
    SIMPLENET_CIFAR100_NAME: 32*32*3,
    INCEPTION_V3_NAME: 299*299*3,
    MOBILENETV2_NAME: 299*299*3,
}

RSLTS_FILENAMES = {'../json_results/FGSMUntargeted.json',
                   '../json_results/FGSMTargeted.json',
                   '../json_results/GenAttack.json',
                   '../json_results/DeepFool.json',
                   '../json_results/BFGS.json',
                   '../json_results/CarliniWagner.json',
                   '../json_results/JSMATargeted.json'}

def bfgs_budget(params, model_name):
    # We do approximately binary steps and for each we calculate gradient iter_max times
    return params[ITER_MAX] * 10

def fgsm_budget(params, model_name):
    # One inference call and one gradient calculation
    return 2 * params[ITER_MAX]

def jsma_budget(params, model_name):
    # iter_max is dependent on nmb of features in dataset
    # we calculate gradient for each of the classes
    iter_max = MODELS_TO_NMB_FEATURES[model_name] * params[MAX_PERTURBATION]
    return iter_max * (MODELS_TO_CLASSES_NMB[model_name] + 1)

def gen_budget(params, model_name):
    # We evaluate specimens two times per generation
    return params[GENERATION_NUMBER] * params[POPULATION_NMB] * 2

def deepfool_budget(params, model_name):
    return params[ITER_MAX] * MODELS_TO_CLASSES_NMB[model_name]

def carlini_wagner_budget(params, model_name):
    #in each optimization iter we do one forward pass and one gradient calculation
    #apar from that we do one forward pass for each binary_iter
    return params[BINARY_ITER] * (2 * params[OPTIMIZATION_ITER] + 1)

BUDGET_ESTIMATION = {
    "wrapper_deepfool": deepfool_budget,
    "wrapped_bfgs": bfgs_budget,
    "wrapped_carlini_wagner": carlini_wagner_budget,
    "wrapped_fgsm": fgsm_budget,
    "wrapped_fgsm_untargeted": fgsm_budget,
    "wrapped_gen_attack": gen_budget,
    "wrapped_jsma": jsma_budget,
}

ATTACK_NAMES_MAP = {
    "wrapper_deepfool": "DeepFool",
    "wrapped_bfgs": "L-BFGS-B",
    "wrapped_carlini_wagner": "Carlini \& Wagner",
    "wrapped_fgsm": "LL-FGSM",
    "wrapped_fgsm_untargeted": "FGSM",
    "wrapped_gen_attack": "GenAttack",
    "wrapped_jsma": "JSMA",
}

ATTACK_NAMES_MAP_MATPLOT = {
    "wrapper_deepfool": "DeepFool",
    "wrapped_bfgs": "L-BFGS-B",
    "wrapped_carlini_wagner": "Carlini & Wagner",
    "wrapped_fgsm": "LL-FGSM",
    "wrapped_fgsm_untargeted": "FGSM",
    "wrapped_gen_attack": "GenAttack",
    "wrapped_jsma": "JSMA",
}



BUDGET_RANGES = [ 10, 100, 1000, 10000, 100000]

def generate_header():
    nmb_metrics = len(PRINTABLE_METRICS)
    header = r'\begin{tabular}{|c|c||' + (r'r|' * nmb_metrics) + r'}' + "\n"
    header += r'\hline' + "\n"

    header += r"\multicolumn{1}{|C|}{Budżet} & \multicolumn{1}{|C||}{Atak} & "
    for i in range(nmb_metrics):
        header += r'\multicolumn{1}{C|}{'
        header += (METRICS_NAME_MAP[PRINTABLE_METRICS[i % nmb_metrics]])
        header += r"}"
        if i < nmb_metrics - 1:
            header += r' & '
    header += r'\\ \hline' + "\n"
    return header

def generate_footer():
    footer = r'\end{tabular}' + "\n"
    return footer

def print_budget_multirow(budget, attacks_dict):
    results = list(map(lambda kvp:kvp[1], attacks_dict.items()))
    results = sorted(results, key=lambda result_dict : result_dict[ACCURACY_KEY], reverse=True)
    row = r"\multirow{" + str(len(results)) + "}{*}{" + str(budget) + "}" + print_results_row(results[0])
    for result_dict in results[1:]:
        row += print_results_row(result_dict)
    row += "\hline \n"
    return row

def print_results_row(results_dict):
    row = " & " + ATTACK_NAMES_MAP[results_dict[ATTACK_NAME_KEY]] + " & "
    for metric in PRINTABLE_METRICS:
        try:
            metric_result = results_dict[metric]
        except TypeError:
            metric_result = -1.0
        if metric != "accuracy_result":
            row += ltx_frmt(metric_result) + " & "
        else:
            row += ltx_prcnt(metric_result) + " & "
    row = row[:-3]  # remove last ampersand
    row += r"\\ \cline{2-" + str(len(PRINTABLE_METRICS)+2) + "}"
    row += "\n"
    return row


def get_data_for_each_range(single_dataset_dict):
    budget_results = {} #key : max_budget, value : (dict of (attack_name : highest_acc_result_dict))
    for i, budget in enumerate(BUDGET_RANGES):
        min_budget = 0 if i==0 else BUDGET_RANGES[i-1]
        budget_results[budget] = get_result_highest_in_budget(single_dataset_dict, budget, min_budget)
    return budget_results

def get_result_highest_in_budget(single_dataset_dict, budget, min_budget):
    highest_in_budget_dict = {}
    for attack, budget_dict in single_dataset_dict.items():
        highest_accuracy = -1.0
        for budget_estimation, results in budget_dict.items():
            is_in_budget = budget_estimation > min_budget and budget_estimation <= budget
            sorted_results = sorted(results, key=lambda result_dict : result_dict[ACCURACY_KEY], reverse=True)
            highest_accuracy_in_budget = sorted_results[0][ACCURACY_KEY]
            if is_in_budget and highest_accuracy_in_budget > highest_accuracy:
                highest_in_budget_dict[attack] = sorted_results[0]
    return highest_in_budget_dict

def load_results_into_dicts():
    ds_dict = {}
    for file_name in  RSLTS_FILENAMES:
        file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, file_name)
        file_name = os.path.abspath(file_name)
        with open(file_name, 'r') as file:
            tmp = json.load(file)
            for model, results in tmp.items():
                ds_name = MODELS_TO_DATASETS[model]
                atck_name = results[0][ATTACK_NAME_KEY]
                budget_function = BUDGET_ESTIMATION[atck_name]
                ds_dict[ds_name] = {} if ds_name not in ds_dict else ds_dict[ds_name]
                ds_dict[ds_name][atck_name] = {} if atck_name not in ds_dict[ds_name] else ds_dict[ds_name][atck_name]
                for result in results:
                    params = result[PARAMETERS_KEY]
                    model_name = result[MODEL_NAME_KEY]
                    budget_estimation = budget_function(params, model_name)
                    ds_dict[ds_name][atck_name][budget_estimation] = [result] if budget_estimation not in ds_dict[ds_name][atck_name] else ds_dict[ds_name][atck_name][budget_estimation] + [result]
    return ds_dict

def print_dataset_table(dataset_name):
    ds_dict = load_results_into_dicts()
    mnist_dict = get_data_for_each_range(ds_dict[dataset_name])
    main_string = generate_header()
    for budget, result in mnist_dict.items():
        main_string += print_budget_multirow(budget, result)
    main_string += generate_footer()
    print(main_string)

ATTACK_NAMES_LIST = [k for k ,v in ATTACK_NAMES_MAP.items()]
ATTACK_COLORS = {
    "wrapper_deepfool": "blue",
    "wrapped_bfgs": "red",
    "wrapped_carlini_wagner": "orange",
    "wrapped_fgsm": "green",
    "wrapped_fgsm_untargeted": "purple",
    "wrapped_gen_attack": "brown",
    "wrapped_jsma": "pink",
}

def generate_plot_data():
    attack_dict_flat = {key : [] for key in ATTACK_NAMES_LIST}
    ds_dict = load_results_into_dicts()
    for dataset_name, attack_dict in ds_dict.items():
        for attack_name, budget_dict in attack_dict.items():
            for budget, result_list in budget_dict.items():
                attack_dict_flat[attack_name] += result_list
    for attack_name, result_list in attack_dict_flat.items():
        tmp_list = list(map(lambda x: (x[ROBUSTNESS_KEY], x[ACCURACY_KEY]), result_list))
        tmp_list = sorted(tmp_list, key= lambda x: ( (1/x[0]) * x[1]), reverse=True)
        if len(tmp_list) >=10:
            tmp_list = tmp_list[:9]
        robustness_list, acc_list = zip(*tmp_list)
        attack_dict_flat[attack_name] = (list(robustness_list), list(acc_list))
    print(attack_dict_flat)
    fig, ax = plt.subplots()
    for attack_key, color in ATTACK_COLORS.items():
        x, y = attack_dict_flat[attack_key]
        scale = 200.0
        ax.scatter(x, y, c=color, s=scale, label=ATTACK_NAMES_MAP_MATPLOT[attack_key],
                   alpha=0.7, edgecolors='none')
    ax.legend()
    ax.grid(True)
    plt.xlabel(r"$\rho_{adw}$")
    plt.ylabel("ACC")
    plt.savefig("../../latex/img/attack_comparison.png")
    plt.show()
