import random
import numpy as np
from tensorflow.python import enable_eager_execution
from models.BaseModels.SequentialModel import SequentialModel
import matplotlib.pyplot as plt


import tensorflow as tf

# TODO: Generalize for all models
# https://arxiv.org/pdf/1412.1897.pdf
# https://arxiv.org/pdf/1504.04909.pdf - tutaj jest opisana strategia ewolucyjna
# https://pdfs.semanticscholar.org/3461/99d0c8fce4bd91be855155bb1fe71844e9ac.pdf - polynomial mutation operator
from models.CIFAR10Models.ConvModel import ConvModel
from models.MNISTModels.LeNet5 import LeNet5
from models.utils.images import show_plot


def evolutionary_attack(model, population_size, iter_max, min=0.0, max=1.0, mutation_chance=0.1, mutation_power=15, use_normal=False, use_gausian=False):
    """
    :type model: SequentialModel
    """
    # population = [] #

    # best_in_class = {} #dictionary of specimen performing best for given class

    performance_map = {}
    specimen_map = {}
    initial_population_size = 1000 * population_size #TODO Ensure that the whole population is initialized or assert that we don't need it in eval_and_insert

    for x in range(initial_population_size):
        new_specimen = np.random.uniform(min, max, model.get_input_shape()) #initialize specimens with random values
        eval_and_insert(performance_map, specimen_map, model, new_specimen)


    for i in range(iter_max):
        if i%100==0: print("generation: " + str(i))
        if i%2000==0: mutation_chance = mutation_chance * 0.5

        current_specimen = random.choice(list(specimen_map.values()))
        mutated_specimen = [mutate_specimen(current_specimen, min, max, mutation_chance, mutation_power=mutation_power) for i in range(2*population_size)]
        for specimen in mutated_specimen:
            eval_and_insert(performance_map, specimen_map, model, specimen)

    for cur_class, specimen in specimen_map.items():
        specimen = tf.expand_dims(tf.convert_to_tensor(specimen, dtype=tf.float32), 0)
        logits = model(specimen)
        show_plot(logits, specimen, model.get_label_names())

def eval_and_insert(performance_map, specimen_map, model, specimen):
    cur_performance = eval_specimen(specimen, model)
    for idx, performance in enumerate(cur_performance):
        if idx in performance_map:
            if cur_performance[idx] > performance_map[idx]:
                performance_map[idx] = cur_performance[idx]
                specimen_map[idx] = specimen
        else:
            performance_map[idx] = cur_performance[idx]
            specimen_map[idx] = specimen


def eval_specimen(specimen, model):
    logits = model(tf.expand_dims(tf.convert_to_tensor(specimen, dtype=tf.float32), 0))
    return tf.squeeze(logits).numpy()

def mutate_specimen(specimen, min, max, mutation_chance, mutation_power):
    new_specimen = np.copy(specimen)
    with np.nditer(new_specimen, op_flags=['readwrite']) as it:
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
    return np.clip(new_specimen, min, max)

def mutate_specimen_gausian(specimen, min, max, mutation_chance):
    specimen = np.copy(specimen)
    std_dev = 0.25 * (max-min)
    with np.nditer(specimen, op_flags=['readwrite']) as it:
        for old_val in it:
            if np.random.uniform() < mutation_chance:
                new_val = np.random.normal(old_val, std_dev)
                old_val[...] = np.clip(new_val, min, max)
    return np.clip(specimen, min, max)


def test_ea_attack():
    # model = DenseModel().load_model_data()
    # model = ConvModel().load_model_data()
    model = LeNet5().load_model_data()
    # model.test()
    evolutionary_attack(model, model.get_number_of_classes(), 10000, mutation_power=15, mutation_chance=0.5, use_normal=False, use_gausian=False)


if __name__ == "__main__":
    enable_eager_execution()
    test_ea_attack()
