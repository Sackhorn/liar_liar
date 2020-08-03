from liar_liar.attacks.deepfool import deepfool_wrapper
from liar_liar.models.base_models.model_names import *
from liar_liar.utils.general_names import *
from liar_liar.utils.test_run_attacks import attack_with_params_dict

params_list = [
    {ITER_MAX: 1},
    {ITER_MAX: 10},
    {ITER_MAX: 100},
]

attack_params = {
    SIMPLENET_CIFAR10_NAME :
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:500, NMB_ELEMENTS_KEY:10}
    },
    SIMPLENET_CIFAR100_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:500, NMB_ELEMENTS_KEY:-1}
    },
    MNIST_TF_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:500, NMB_ELEMENTS_KEY:-1}
    },
    LE_NET_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1000, NMB_ELEMENTS_KEY:-1}
    },
    INCEPTION_V3_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1000}
    },
    MOBILENETV2_NAME:
    {
        PARAMETERS_KEY : params_list,
        DATASET_KEY: {BATCHES_KEY:1, NMB_ELEMENTS_KEY:1000}
    },
}

if __name__ == "__main__":
    # import tensorflow as tf
    # import os
    # from liar_liar.base_models.data_provider import DataProvider
    # from datetime import datetime
    # date_string = datetime.today().strftime("%d_%m_%Y_%H_%M")
    # logdir = os.path.join(DataProvider.ROOT_DIR, DataProvider.TENSORBOARD_NAME, deepfool_wrapper.__name__, date_string)
    # with tf.profiler.experimental.Profile(logdir):
    attack_with_params_dict(attack_params, deepfool_wrapper, show_plot=True, targeted=False)
