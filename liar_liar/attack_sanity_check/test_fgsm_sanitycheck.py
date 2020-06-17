from unittest import TestCase

from liar_liar.attacks.fgsm import fgsm_targeted_wrapper, fgsm_untargeted_wrapper
from liar_liar.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict


attack_params = {
    CIFAR_10_CONV_NAME :
    {
        PARAMETERS_KEY : [{ITER_MAX: 1000, EPS: 0.0001}, {ITER_MAX: 10, EPS: 0.1}, ],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    CIFAR_100_CONV_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}, {ITER_MAX: 10, EPS: 0.1}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    MNIST_CONV_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}, {ITER_MAX: 10, EPS: 0.1}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
    MNIST_DENSE_NAME:
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
    RESNET_NAME:
    {
        PARAMETERS_KEY : [{ITER_MAX: 10, EPS: 0.01}, {ITER_MAX: 10, EPS: 0.1}],
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:1}
    },
}

class FGSMSanityCheck(TestCase):
    def test_sanitycheck(self):
        fgsm_targeted_wrapper.__name__ += "_sanity_check"
        fgsm_untargeted_wrapper.__name__ += "_sanity_check"
        attack_with_params_dict(attack_params, fgsm_targeted_wrapper, show_plot=False, targeted=True)
        attack_with_params_dict(attack_params, fgsm_untargeted_wrapper, show_plot=False, targeted=False)
