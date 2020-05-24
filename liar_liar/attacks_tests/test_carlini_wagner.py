from liar_liar.attacks.c_and_w import carlini_wagner_wrapper
from liar_liar.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

attack_params = {
    CIFAR_10_CONV_NAME :
    {
        PARAMETERS_KEY : [{OPTIMIZATION_ITER:1000, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    CIFAR_100_CONV_NAME:
    {
        PARAMETERS_KEY : [{OPTIMIZATION_ITER:1000, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    MNIST_CONV_NAME:
    {
        PARAMETERS_KEY : [{OPTIMIZATION_ITER:1000, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    MNIST_DENSE_NAME:
    {
        PARAMETERS_KEY : [{OPTIMIZATION_ITER:1000, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : [{OPTIMIZATION_ITER:1000, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : [{OPTIMIZATION_ITER:1000, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
    RESNET_NAME:
    {
        PARAMETERS_KEY : [{OPTIMIZATION_ITER:1000, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
}

if __name__ == "__main__":
    attack_with_params_dict(attack_params, carlini_wagner_wrapper, show_plot=True, targeted=True)
