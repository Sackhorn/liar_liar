from liar_liar.attacks.c_and_w import carlini_wagner_wrapper
from liar_liar.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

params_list = [
    {OPTIMIZATION_ITER:1000, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},
    {OPTIMIZATION_ITER:100, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},
    {OPTIMIZATION_ITER:10, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},
    {OPTIMIZATION_ITER:1, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},

    {OPTIMIZATION_ITER:1000, BINARY_ITER:20, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},
    {OPTIMIZATION_ITER:100, BINARY_ITER:20, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},
    {OPTIMIZATION_ITER:10, BINARY_ITER:20, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},
    {OPTIMIZATION_ITER:1, BINARY_ITER:20, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},

    {OPTIMIZATION_ITER:1000, BINARY_ITER:20, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.5},
    {OPTIMIZATION_ITER:100, BINARY_ITER:20, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.5},
    {OPTIMIZATION_ITER:10, BINARY_ITER:20, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.5},
    {OPTIMIZATION_ITER:1, BINARY_ITER:20, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.5},

]

attack_params = {
    SIMPLENET_CIFAR10_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    SIMPLENET_CIFAR100_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    MNIST_TF_NAME:
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
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:500}
    },
    MOBILENETV2_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:500}
    },
}

if __name__ == "__main__":
    attack_with_params_dict(attack_params, carlini_wagner_wrapper, show_plot=True, targeted=True)
