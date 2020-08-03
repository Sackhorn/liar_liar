from unittest import TestCase

from liar_liar.models.base_models.model_names import *
from liar_liar.attacks.c_and_w import carlini_wagner_wrapper
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

attack_params = {
    SIMPLENET_CIFAR10_NAME :
    {
        PARAMETERS_KEY : [
            {OPTIMIZATION_ITER:2, BINARY_ITER:2, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},
            {OPTIMIZATION_ITER:1, BINARY_ITER:1, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    SIMPLENET_CIFAR100_NAME:
    {
        PARAMETERS_KEY : [
            {OPTIMIZATION_ITER: 2, BINARY_ITER: 2, C_HIGH: 100.0, C_LOW: 0.0, KAPPA: 0.0},
            {OPTIMIZATION_ITER: 1, BINARY_ITER: 1, C_HIGH: 100.0, C_LOW: 0.0, KAPPA: 0.0}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    MNIST_TF_NAME:
    {
        PARAMETERS_KEY : [
            {OPTIMIZATION_ITER: 2, BINARY_ITER: 2, C_HIGH: 100.0, C_LOW: 0.0, KAPPA: 0.0},
            {OPTIMIZATION_ITER: 1, BINARY_ITER: 1, C_HIGH: 100.0, C_LOW: 0.0, KAPPA: 0.0}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : [
            {OPTIMIZATION_ITER: 2, BINARY_ITER: 2, C_HIGH: 100.0, C_LOW: 0.0, KAPPA: 0.0},
            {OPTIMIZATION_ITER: 1, BINARY_ITER: 1, C_HIGH: 100.0, C_LOW: 0.0, KAPPA: 0.0}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : [
            {OPTIMIZATION_ITER: 2, BINARY_ITER: 2, C_HIGH: 100.0, C_LOW: 0.0, KAPPA: 0.0},
            {OPTIMIZATION_ITER: 1, BINARY_ITER: 1, C_HIGH: 100.0, C_LOW: 0.0, KAPPA: 0.0}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    MOBILENETV2_NAME:
    {
        PARAMETERS_KEY : [
            {OPTIMIZATION_ITER: 2, BINARY_ITER: 2, C_HIGH: 100.0, C_LOW: 0.0, KAPPA: 0.0},
            {OPTIMIZATION_ITER: 1, BINARY_ITER: 1, C_HIGH: 100.0, C_LOW: 0.0, KAPPA: 0.0}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
}

class CarliniWagnerSanityCheck(TestCase):
    def test_sanitycheck(self):
        carlini_wagner_wrapper.__name__ = carlini_wagner_wrapper.__name__ + "_sanity_check"
        attack_with_params_dict(attack_params, carlini_wagner_wrapper, show_plot=False, targeted=True)
