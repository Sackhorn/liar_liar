from liar_liar.attacks.gen_attack import gen_attack_wrapper
from liar_liar.models.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

gen_attack_params = [
    {GENERATION_NUMBER:1000, POPULATION_NMB:5, DELTA:0.05, MUTATION_PROBABILITY:0.05},
]

attack_params = {
    SIMPLENET_CIFAR10_NAME :
    {
        PARAMETERS_KEY : gen_attack_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:1}
    },
    SIMPLENET_CIFAR100_NAME:
    {
        PARAMETERS_KEY : gen_attack_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:1}
    },
    MNIST_TF_NAME:
    {
        PARAMETERS_KEY : gen_attack_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:1}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : gen_attack_params,
        DATASET_KEY: {BATCHES_KEY:100, NMB_ELEMENTS_KEY:1}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : gen_attack_params,
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
    MOBILENETV2_NAME:
    {
        PARAMETERS_KEY : gen_attack_params,
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
}

if __name__ == "__main__":
    import tensorflow as tf
    tf.profiler.experimental.server.start(6009)
    attack_with_params_dict(attack_params, gen_attack_wrapper, show_plot=False, targeted=True)
