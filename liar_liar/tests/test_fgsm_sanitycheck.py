from unittest import TestCase

from liar_liar.attacks.fgsm import FGSMTargeted, FGSMUntargeted
from liar_liar.models.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

attack_params = {
    SIMPLENET_CIFAR10_NAME :
    {
        PARAMETERS_KEY : [{ITER_MAX: 1000, EPS: 0.0001}, {ITER_MAX: 10, EPS: 0.1}, ],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    SIMPLENET_CIFAR100_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}, {ITER_MAX: 10, EPS: 0.1}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    MNIST_TF_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}, {ITER_MAX: 10, EPS: 0.1}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}, {ITER_MAX: 10, EPS: 0.1}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}, {ITER_MAX: 10, EPS: 0.1}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    MOBILENETV2_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}, {ITER_MAX: 10, EPS: 0.1}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
}

class FGSMSanityCheck(TestCase):
    def test_sanitycheck(self):
        FGSMTargeted.__name__ += "_sanity_check"
        FGSMUntargeted.__name__ += "_sanity_check"
        attack_with_params_dict(attack_params, FGSMTargeted, show_plot=False, targeted=True)
        attack_with_params_dict(attack_params, FGSMUntargeted, show_plot=False, targeted=False)
