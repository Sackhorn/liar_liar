import tensorflow as tf
import tensorflow_datasets as tfds
from math import floor

from models.Attacks.jsma import jsma, jsma_plus_increasing

import models.BaseModels.DataProvider
from models.CIFAR10Models.ConvModel import ConvModel


def run_targeted_atacks(attack, model, dataset_percentage):
    """

    :type model: models.BaseModels.SequentialModel
    """
    succesful_attacks = 0
    dataset = model.get_dataset(tfds.Split.TEST, batch_size=1, shuffle=1)
    test_samples_nmb = floor(model.get_info().splits['test'].num_examples * dataset_percentage)
    for data_sample in dataset.take(test_samples_nmb):
        image, true_class = data_sample
        true_class = true_class.numpy().squeeze().argmax()
        target_class = true_class - 1 % model.get_number_of_classes()
        if true_class != model(image).numpy().squeeze().argmax():
            continue
        image, return_class = attack.run_attack(model, data_sample, target_class)
        succesful_attacks += 1 if return_class == target_class else 0
        print(succesful_attacks)
    failed_attacks = test_samples_nmb - succesful_attacks
    print("success_ratio is {0.2f}".format((succesful_attacks/test_samples_nmb)*100))
    return succesful_attacks, failed_attacks


run_targeted_atacks(jsma_plus_increasing, ConvModel().load_model_data(), 0.2)