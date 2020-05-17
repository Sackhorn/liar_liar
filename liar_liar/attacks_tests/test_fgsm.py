from liar_liar.attacks.fgsm import fgsm_targeted_wrapper
from liar_liar.utils.test_run_attacks import run_test, attack_with_params_dict
from liar_liar.utils.general_names import *
from liar_liar.base_models.model_names import *


ITER_MAX = "iter_max"
EPS = "eps"

attack_params = {
    CIFAR_10_CONV_NAME : 
    {
        PARAMETERS_KEY : [{ITER_MAX: 1000, EPS: 0.0001}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:10}
    },
    CIFAR_100_CONV_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:10}
    },
    MNIST_CONV_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:10}
    },
    MNIST_DENSE_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:10}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:10}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
    RESNET_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
}

if __name__ == "__main__":
    attack_with_params_dict(attack_params, fgsm_targeted_wrapper, show_plot=True, targeted=True)
