from liar_liar.attacks.jsma import jsma_untargeted_wrapper, jsma_targeted_wrapper
from liar_liar.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

MAX_PER

MAX_PERTURBATION = "max_perturbation" #=0.1,
THETA = "theta" #=1,
IS_INCREASING = "is_increasing" #=True,
USE_LOGITS = "use_logits" #=False

attack_params = {
    CIFAR_10_CONV_NAME :
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    CIFAR_100_CONV_NAME:
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    MNIST_CONV_NAME:
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    MNIST_DENSE_NAME:
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    # INCEPTION_V3_NAME:
    # {
    #     PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
    #     DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    # },
    # RESNET_NAME:
    # {
    #     PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
    #     DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    # },
}

if __name__ == "__main__":
    attack_with_params_dict(attack_params, jsma_untargeted_wrapper, show_plot=False, targeted=False)
    attack_with_params_dict(attack_params, jsma_targeted_wrapper, show_plot=True, targeted=True)
