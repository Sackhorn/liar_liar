from liar_liar.attacks.bfgs import bfgs_wrapper
from liar_liar.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict


bfgs_params = [
    {ITER_MAX:100},
    {ITER_MAX:1000},
    # {ITER_MAX:10000}
]


attack_params = {
    SIMPLENET_CIFAR10_NAME :
    {
        PARAMETERS_KEY : bfgs_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:-1}
    },
    SIMPLENET_CIFAR100_NAME:
    {
        PARAMETERS_KEY : bfgs_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:-1}
    },
    MNIST_TF_NAME:
    {
        PARAMETERS_KEY : bfgs_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:-1}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : bfgs_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:-1}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : bfgs_params,
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
    MOBILENETV2_NAME:
    {
        PARAMETERS_KEY : bfgs_params,
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
}

if __name__ == "__main__":
    attack_with_params_dict(attack_params, bfgs_wrapper, show_plot=False, targeted=True)
