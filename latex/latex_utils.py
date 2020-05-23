import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


CIFAR_10_CONV_NAME = "cifar10_conv_model"
CIFAR_100_CONV_NAME = "cifar100_conv_model"
INCEPTION_V3_NAME = "imagenet_v3"
RESNET_NAME = "resnet50v2"
MNIST_CONV_NAME = "mnist_conv_model"
MNIST_DENSE_NAME = "mnist_dense_model"
LE_NET_NAME = "le_net_model"

METRICS_NAME_MAP = {
    "accuracy_result" : "Accuracy",
    "L2_median" : "L2 Median",
    "L2_average" : "L2 Average",
    "average_time_per_sample" : "Time per sample",
}

MODEL_NAME_MAP = {
CIFAR_10_CONV_NAME : "CIFAR-10",
CIFAR_100_CONV_NAME : "CIFAR-10",
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

PARAMS_SETS_IN_TABLE = [{"eps":0.0001, "iter_max":1000}, {"eps":0.001, "iter_max":10}]
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

def import_and_print(file_name):
    results = None
    with open('../json_results/' + file_name, 'r') as file:
        results = json.load(file)
    results = filter_for_params(results, PARAMS_SETS_IN_TABLE)
    #we cell for each metric in each param set
    nmb_columns = len(PARAMS_SETS_IN_TABLE) * len(PRINTABLE_METRICS)
    row_columns = r'\begin{tabular}{|c||' + (r'c|' * nmb_columns) + r'}'
    print(row_columns)
    # print(r'\centered')
    print(r'\hline')

    # a cell for each parameter set
    header_row = ""
    for i in range(len(PARAMS_SETS_IN_TABLE)):
        header_row += r"& \multicolumn{" + str(len(PRINTABLE_METRICS)) + r"}{c|}{" + r"eps={}, i={}}}".format(PARAMS_SETS_IN_TABLE[i]["eps"], PARAMS_SETS_IN_TABLE[i]["iter_max"])
    header_row += r" \\ \hline"
    print(header_row)

    # Create column names
    header_row = r"Model Name & "
    for i in range(nmb_columns):
        header_row +=(METRICS_NAME_MAP[PRINTABLE_METRICS[i%len(PRINTABLE_METRICS)]])
        if i < nmb_columns -1:
            header_row += r' & '
    header_row += r'\\ \hline'
    print(header_row)

    #Fill rows with data
    for model_name, results_list in results.items():
        row = str(MODEL_NAME_MAP[model_name]) + " & "
        for result in  results_list:
            for metric in PRINTABLE_METRICS:
                if metric!="accuracy_result":
                    row += ltx_frmt(result[metric]) + " & "
                else:
                    row += ltx_prcnt(result[metric]) + " & "
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

    print(r"\begin{center}")
    print(r"\includegraphics[width=0.85\textwidth]{fgsm_heat_map.pdf}")
    print(r"\end{center}")





