from liar_liar.attacks.jsma import jsma_untargeted_wrapper, jsma_targeted_wrapper
from liar_liar.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict



attack_params = {
    SIMPLENET_CIFAR10_NAME :
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:10, NMB_ELEMENTS_KEY:100}
    },
    SIMPLENET_CIFAR100_NAME:
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:10, NMB_ELEMENTS_KEY:100}
    },
    MNIST_TF_NAME:
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:10, NMB_ELEMENTS_KEY:100}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : [{MAX_PERTURBATION:0.1, THETA:1, IS_INCREASING:True, USE_LOGITS: False}],
        DATASET_KEY: {BATCHES_KEY:10, NMB_ELEMENTS_KEY:100}
    },
}

if __name__ == "__main__":
    # TODO: Fix untargeted version of jsma
    # attack_with_params_dict(attack_params, jsma_untargeted_wrapper, show_plot=False, targeted=False)
    attack_with_params_dict(attack_params, jsma_targeted_wrapper, show_plot=False, targeted=True)
