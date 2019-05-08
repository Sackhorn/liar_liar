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
from models.utils.images import show_plot


def evolutionary_attack(model, population_size, iter_max, min=0.0, max=1.0, mutation_chance=0.1, mutation_power=15):
    """
    :type model: SequentialModel
    """
    population = [] #
    best_in_class = {} #dictionary of specimen performing best for given class

    for x in range(population_size):
        population.append(np.random.uniform(min, max, model.get_input_shape())) #initialize specimens with random values

    for i in range(iter_max):
        if i%100==0: print(i)
        current_specimen = random.choice(population)
        mutated_specimen = mutate_specimen(current_specimen, min, max, mutation_chance, mutation_power)
        logits = model(tf.expand_dims(tf.convert_to_tensor(mutated_specimen, dtype=tf.float32), 0), get_raw=True)
        best_class = tf.argmax(tf.squeeze(logits)).numpy()
        certainty = tf.reduce_max(tf.squeeze(logits)).numpy()
        if best_class in best_in_class:
            _, best_certainty = best_in_class[best_class]
            if certainty > best_certainty:
                best_in_class[best_class] = (mutated_specimen, certainty)
                population.append(mutated_specimen)
        else:
            best_in_class[best_class] = (mutated_specimen, certainty)

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


def test_ea_attack():
    model = DenseModel()
    model.load_model_data()
    model.test()
    # for i in range(10):
    #     numpy = np.random.uniform(low=0.0, high=1.0, size=model.get_input_shape())
    #     plt.imshow(np.squeeze(numpy), cmap='gray')
    #     plt.show()
    #     tensor = tf.expand_dims(tf.convert_to_tensor(numpy, dtype=tf.float32), 0)
    #     logits = model(tensor, get_raw=True)
    #     show_plot(logits, tensor)
    evolutionary_attack(model, 1000, 10000, mutation_chance=0.5)

    # dict = {}
    # for datasample in model.get_dataset(Split.TRAIN, batch_size=1):
    #     image, label = datasample
    #     numb = tf.argmax(tf.squeeze(label)).numpy().astype(int)
    #     if numb not in dict:
    #         dict[numb] = 1
    #     else:
    #         dict[numb] = dict[numb] + 1
    #
    # print (dict)

if __name__ == "__main__":
    enable_eager_execution()
    test_ea_attack()
