from liar_liar.attacks.bfgs import BFGS
from liar_liar.models.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict


bfgs_params = [
    {ITER_MAX:100},
    {ITER_MAX:1000},
]


attack_params = {
    SIMPLENET_CIFAR10_NAME :
    {
        PARAMETERS_KEY : bfgs_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:10}
    },
    SIMPLENET_CIFAR100_NAME:
    {
        PARAMETERS_KEY : bfgs_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:10}
    },
    MNIST_TF_NAME:
    {
        PARAMETERS_KEY : bfgs_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:10}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : bfgs_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:10}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : bfgs_params,
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:500}
    },
    MOBILENETV2_NAME:
    {
        PARAMETERS_KEY : bfgs_params,
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:500}
    },
}

if __name__ == "__main__":
    attack_with_params_dict(attack_params, BFGS, show_plot=False, targeted=True)
