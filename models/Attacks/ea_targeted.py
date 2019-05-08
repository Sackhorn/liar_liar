import random

import numpy as np
from tensorflow.python import enable_eager_execution
import matplotlib.pyplot as plt
from tensorflow_datasets import Split

from models.CIFAR10Models.ConvModel import ConvModel
from models.MNISTModels.DenseModel import DenseModel
from models.BaseModels.SequentialModel import SequentialModel

import tensorflow as tf

# TODO: Generalize for all models
# https://arxiv.org/pdf/1412.1897.pdf
# https://arxiv.org/pdf/1504.04909.pdf - tutaj jest opisana strategia ewolucyjna
# https://pdfs.semanticscholar.org/3461/99d0c8fce4bd91be855155bb1fe71844e9ac.pdf - polynomial mutation operator
from models.utils.images import show_plot


def evolutionary_attack(model, population_size, generation, target_class, min=0.0, max=1.0, mutation_chance=0.1, mutation_power=15):
    """
    :type model: SequentialModel
    """
    population = [] #
    best = (None, 0.0)

    for x in range(population_size):
        population.append(np.random.uniform(min, max, model.get_input_shape())) #initialize specimens with random values

    for i in range(generation):
        for specimen in population:
            logits = model(tf.expand_dims(tf.convert_to_tensor(specimen, dtype=tf.float32), 0))
            certainty = tf.squeeze(logits).numpy()[target_class]
            if certainty > best[1]:
                best = (specimen, certainty)

        population = [mutate_specimen(best[0], min, max, mutation_chance, mutation_power) for _ in range(population_size)]


    specimen, _ = best
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


def test_ea_attack():
    model = DenseModel()
    model.load_model_data()
    evolutionary_attack(model, 10, 200, 9, mutation_power=20)

if __name__ == "__main__":
    enable_eager_execution()
    test_ea_attack()
