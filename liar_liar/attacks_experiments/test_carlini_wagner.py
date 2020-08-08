from liar_liar.attacks.c_and_w import carlini_wagner_wrapper
from liar_liar.models.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

carlini_wagner_params = [
    {OPTIMIZATION_ITER:1000, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},
    {OPTIMIZATION_ITER:100, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},
    {OPTIMIZATION_ITER:10, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},
    {OPTIMIZATION_ITER:1, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},
]

attack_params = {
    SIMPLENET_CIFAR10_NAME:
    {
        PARAMETERS_KEY : carlini_wagner_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:10}
    },
    SIMPLENET_CIFAR100_NAME:
    {
        PARAMETERS_KEY : carlini_wagner_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:10}
    },
    MNIST_TF_NAME:
    {
        PARAMETERS_KEY : carlini_wagner_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:10}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : carlini_wagner_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:10}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : carlini_wagner_params,
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
    MOBILENETV2_NAME:
    {
        PARAMETERS_KEY : carlini_wagner_params,
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
}

if __name__ == "__main__":
    attack_with_params_dict(attack_params, carlini_wagner_wrapper, show_plot=False, targeted=True)
