from liar_liar.attacks.fgsm import fgsm_targeted_wrapper, fgsm_untargeted_wrapper
from liar_liar.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

# possible_iters = [1000, 100, 10, 1]
# possible_eps = [0.1, 0.01, 0.001, 0.0001]
#
# params_list = []
# for i_max in possible_iters:
#     for eps in possible_eps:
#         params_list.append({ITER_MAX:i_max, EPS:eps})

params_list = [
    {EPS: 0.1, ITER_MAX: 1},
    {EPS: 0.1, ITER_MAX: 10},
    {EPS: 0.5, ITER_MAX: 1},
    {EPS: 0.5, ITER_MAX: 10},
    {EPS: 0.01, ITER_MAX: 1},
    {EPS: 0.01, ITER_MAX: 10},
    {EPS: 0.01, ITER_MAX: 100},
    {EPS: 0.05, ITER_MAX: 10},
    {EPS: 0.05, ITER_MAX: 100},
    {EPS: 0.001, ITER_MAX: 1},
    {EPS: 0.001, ITER_MAX: 10},
    {EPS: 0.001, ITER_MAX: 100},
    {EPS: 0.001, ITER_MAX: 1000},
    {EPS: 0.005, ITER_MAX: 1},
    {EPS: 0.005, ITER_MAX: 10},
    {EPS: 0.005, ITER_MAX: 100},
    {EPS: 0.005, ITER_MAX: 1000},
    {EPS: 0.0001, ITER_MAX: 1},
    {EPS: 0.0001, ITER_MAX: 10},
    {EPS: 0.0001, ITER_MAX: 100},
    {EPS: 0.0001, ITER_MAX: 1000},
    {EPS: 0.0005, ITER_MAX: 1},
    {EPS: 0.0005, ITER_MAX: 10},
    {EPS: 0.0005, ITER_MAX: 100},
    {EPS: 0.0005, ITER_MAX: 1000},
]

attack_params = {
    CIFAR_10_CONV_NAME :
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    CIFAR_100_CONV_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    MNIST_CONV_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    MNIST_DENSE_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
    RESNET_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:2, NMB_ELEMENTS_KEY:50}
    },
}

if __name__ == "__main__":
    attack_with_params_dict(attack_params, fgsm_targeted_wrapper, show_plot=False, targeted=True)
    attack_with_params_dict(attack_params, fgsm_untargeted_wrapper, show_plot=False, targeted=False)