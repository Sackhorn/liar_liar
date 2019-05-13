import random

import numpy as np
from tensorflow.python import enable_eager_execution
import matplotlib.pyplot as plt
from tensorflow_datasets import Split

from models.MNISTModels.ConvModel import ConvModel
from models.MNISTModels.DenseModel import DenseModel
from models.BaseModels.SequentialModel import SequentialModel

import tensorflow as tf

# TODO: Generalize for all models
# https://arxiv.org/pdf/1412.1897.pdf
# https://arxiv.org/pdf/1504.04909.pdf - tutaj jest opisana strategia ewolucyjna
# https://pdfs.semanticscholar.org/3461/99d0c8fce4bd91be855155bb1fe71844e9ac.pdf - polynomial mutation operator
from models.MNISTModels.LeNet5 import LeNet5
from models.utils.images import show_plot


def evolutionary_attack(model, population_size, iter_max, min=0.0, max=1.0, mutation_chance=0.1, mutation_power=20, use_normal=False, use_gausian=False):
    """
    :type model: SequentialModel
    """
    population = [] #
    best_in_class = {} #dictionary of specimen performing best for given class

    for x in range(population_size):
        if use_normal:
            population.append(np.random.normal(0.0, 0.3, model.get_input_shape())) #initialize specimens with random values
        else:
            population.append(np.random.uniform(min, max, model.get_input_shape())) #initialize specimens with random values

    for i in range(iter_max):
        if i%100==0: print(i)
        current_specimen = random.choice(population)
        if use_gausian:
            mutated_specimen = mutate_specimen_gausian(current_specimen, min, max, mutation_chance, mutation_power)
        else:
            mutated_specimen = mutate_specimen(current_specimen, min, max, mutation_chance, mutation_power)
        logits = model(tf.expand_dims(tf.convert_to_tensor(mutated_specimen, dtype=tf.float32), 0))

        certainties_per_class = tf.squeeze(logits).numpy()

        for cur_class in range(len(certainties_per_class)):
            if cur_class in best_in_class:
                _, best_certainty = best_in_class[cur_class]
                if certainties_per_class[cur_class] > best_certainty:
                    best_in_class[cur_class] = (mutated_specimen, certainties_per_class[cur_class])
                    population.append(mutated_specimen)
            else:
                best_in_class[cur_class] = (mutated_specimen, certainties_per_class[cur_class])

    for _, specimen_tuple in best_in_class.items():
        specimen, _ = specimen_tuple
        specimen = tf.expand_dims(tf.convert_to_tensor(specimen, dtype=tf.float32), 0)
        logits = model(specimen)
        show_plot(logits, specimen, model.get_label_names())

def mutate_specimen(specimen, min, max, mutation_chance, mutation_power=15):
    specimen = np.copy(specimen)
    with np.nditer(specimen, op_flags=['readwrite']) as it:
        for old_val in it:
            if np.random.uniform() < mutation_chance:
                u = np.random.uniform()
                if u <= 0.5:
                    delta = ((2.0 * u) ** (1.0 / (mutation_power))) - 1.0
                    new_val = old_val + delta * (old_val - min)
                else:
                    delta = 1.0 - (2 * (1 - u))** (1 / (1 + mutation_power))
                    new_val = old_val + delta * (max - old_val)
                old_val[...] = new_val
    return specimen

def mutate_specimen_gausian(specimen, min, max, mutation_chance, mutation_power=15):
    specimen = np.copy(specimen)
    std_dev = 0.5 * (max-min)
    with np.nditer(specimen, op_flags=['readwrite']) as it:
        for old_val in it:
            if np.random.uniform() < mutation_chance:
                new_val = np.random.normal(old_val, std_dev)
                old_val[...] = np.clip(new_val, min, max)
    return specimen


def test_ea_attack():
    model = DenseModel()
    model.load_model_data()
    model.test()
    evolutionary_attack(model, 100, 2000, use_normal=True, use_gausian=False)


if __name__ == "__main__":
    enable_eager_execution()
    test_ea_attack()
