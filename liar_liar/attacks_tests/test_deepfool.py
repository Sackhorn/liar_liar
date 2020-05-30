from liar_liar.attacks.deepfool import deepfool_wrapper, deepfool_wrapper_map
from liar_liar.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

params_list = [
    {ITER_MAX: 1},
    {ITER_MAX: 10},
    {ITER_MAX: 100},
    {ITER_MAX: 1000},
    {ITER_MAX: 10000},
]

attack_params = {
    CIFAR_10_CONV_NAME :
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    CIFAR_100_CONV_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    MNIST_CONV_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:5000, NMB_ELEMENTS_KEY:-1}
    },
    MNIST_DENSE_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1000}
    },
    RESNET_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1000}
    },
}

if __name__ == "__main__":
    attack_with_params_dict(attack_params, deepfool_wrapper_map, show_plot=False, targeted=False)