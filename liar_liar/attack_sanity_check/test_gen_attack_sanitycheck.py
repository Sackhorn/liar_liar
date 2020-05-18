from unittest import TestCase

from liar_liar.attacks.gen_attack import gen_attack_wrapper
from liar_liar.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

GENERATION_NUMBER = "generation_nmb"
POPULATION_NMB = "population_nmb"
MUTATION_PROBABILITY = "mutation_probability"
DELTA = "delta"


attack_params = {
    CIFAR_10_CONV_NAME :
    {
        PARAMETERS_KEY : [{GENERATION_NUMBER:1000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    CIFAR_100_CONV_NAME:
    {
        PARAMETERS_KEY : [{GENERATION_NUMBER:1000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    MNIST_CONV_NAME:
    {
        PARAMETERS_KEY : [{GENERATION_NUMBER:1000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
    MNIST_DENSE_NAME:
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
    RESNET_NAME:
    {
        PARAMETERS_KEY : [{GENERATION_NUMBER:1000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}],
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1}
    },
}
class GenAttackSanityCheck(TestCase):
    def test_sanitycheck(self):
        gen_attack_wrapper.__name__ = gen_attack_wrapper.__name__ + "_sanity_check"
        attack_with_params_dict(attack_params, gen_attack_wrapper, show_plot=False, targeted=True)
