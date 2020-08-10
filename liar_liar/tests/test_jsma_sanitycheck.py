from unittest import TestCase

from liar_liar.attacks.jsma import JSMATargeted, JSMAUntargeted
from liar_liar.models.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

attack_params = {
    SIMPLENET_CIFAR10_NAME :
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    SIMPLENET_CIFAR100_NAME:
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    MNIST_TF_NAME:
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
}

class JSMASanityCheck(TestCase):
    def test_sanitycheck(self):
        JSMAUntargeted.__name__ += "_sanity_check"
        JSMATargeted.__name__ += "_sanity_check"
        # TODO: Fix untargeted version of jsma
        # attack_with_params_dict(attack_params, JSMAUntargeted, show_plot=False, targeted=False)
        attack_with_params_dict(attack_params, JSMATargeted, show_plot=False, targeted=True)
