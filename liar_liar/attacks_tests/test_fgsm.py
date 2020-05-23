from liar_liar.attacks.fgsm import fgsm_targeted_wrapper, fgsm_untargeted_wrapper
from liar_liar.utils.test_run_attacks import run_test, attack_with_params_dict
from liar_liar.utils.general_names import *
from liar_liar.base_models.model_names import *


ITER_MAX = "iter_max"
EPS = "eps"

possible_iters = [1000, 100, 10, 1]
possible_eps = [0.1, 0.01, 0.001, 0.0001]

params_list = []
for i_max in possible_iters:
    for eps in possible_eps:
        params_list.append({ITER_MAX:i_max, EPS:eps})

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
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
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
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
    RESNET_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
}

if __name__ == "__main__":
    attack_with_params_dict(attack_params, fgsm_targeted_wrapper, show_plot=False, targeted=True)
    attack_with_params_dict(attack_params, fgsm_untargeted_wrapper, show_plot=False, targeted=False)
