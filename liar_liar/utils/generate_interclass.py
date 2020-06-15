from liar_liar.attacks.fgsm import fgsm_targeted_wrapper
from liar_liar.attacks.gen_attack import gen_attack_wrapper
from liar_liar.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import generate_targeted_attack_grid

llfgsm_params = {
# MNIST_CONV_NAME:
#     {
#         PARAMETERS_KEY: [{ITER_MAX: 1000, EPS: 0.0005}],
#     },
# CIFAR_10_CONV_NAME:
#     {
#         PARAMETERS_KEY: [{ITER_MAX: 100, EPS: 0.0005}],
#     },
# CIFAR_100_CONV_NAME:
#     {
#         PARAMETERS_KEY: [{ITER_MAX: 100, EPS: 0.0001}],
#     },
INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY: [{ITER_MAX: 100, EPS: 0.0001}],
    },

}

generate_targeted_attack_grid(llfgsm_params, fgsm_targeted_wrapper, "../../latex/img/llfgsm_interclass", retries=50)


gen_attack_params = [{GENERATION_NUMBER:10000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}]
genattack_params_models = {
    CIFAR_10_CONV_NAME :
    {
        PARAMETERS_KEY : gen_attack_params,
    },
    # CIFAR_100_CONV_NAME:
    # {
    #     PARAMETERS_KEY : gen_attack_params,
    # },
    # MNIST_CONV_NAME:
    # {
    #     PARAMETERS_KEY : gen_attack_params,
    # },
    # INCEPTION_V3_NAME:
    # {
    #     PARAMETERS_KEY : gen_attack_params,
    # },
}

# generate_targeted_attack_grid(genattack_params_models,
#                               gen_attack_wrapper,
#                               "../../latex/img/genattack_interclass",
#                               retries=5)