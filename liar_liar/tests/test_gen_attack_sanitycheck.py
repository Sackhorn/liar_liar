from unittest import TestCase

from liar_liar.attacks.gen_attack import GenAttack
from liar_liar.models.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

attack_params = {
    SIMPLENET_CIFAR10_NAME :
    {
        PARAMETERS_KEY : [{GENERATION_NUMBER:1000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    SIMPLENET_CIFAR100_NAME:
    {
        PARAMETERS_KEY : [{GENERATION_NUMBER:1000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    MNIST_TF_NAME:
    {
        PARAMETERS_KEY : [{GENERATION_NUMBER:1000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : [{GENERATION_NUMBER:1000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : [{GENERATION_NUMBER:1000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    MOBILENETV2_NAME:
    {
        PARAMETERS_KEY : [{GENERATION_NUMBER:1000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
}
class GenAttackSanityCheck(TestCase):
    def test_sanitycheck(self):
        GenAttack.__name__ += "_sanity_check"
        attack_with_params_dict(attack_params, GenAttack, show_plot=False, targeted=True)
