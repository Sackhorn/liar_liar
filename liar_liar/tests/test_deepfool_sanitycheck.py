from unittest import TestCase

from liar_liar.models.base_models.model_names import *
from liar_liar.attacks.deepfool import deepfool_wrapper_map
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

attack_params = {
    SIMPLENET_CIFAR10_NAME :
    {
        PARAMETERS_KEY : [{ITER_MAX: 2}, {ITER_MAX: 1}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    SIMPLENET_CIFAR100_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 2}, {ITER_MAX: 1}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    MNIST_TF_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 2}, {ITER_MAX: 1}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 2}, {ITER_MAX: 1}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 2}, {ITER_MAX: 1}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    MOBILENETV2_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 2}, {ITER_MAX: 1}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
}
class DeepFoolSanityCheck(TestCase):
    def test_sanitycheck(self):
        deepfool_wrapper_map.__name__ += "_sanity_check"
        attack_with_params_dict(attack_params, deepfool_wrapper_map, show_plot=False, targeted=False)
