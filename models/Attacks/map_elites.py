#This is the implementation of method given in this paper
# https://arxiv.org/pdf/1412.1897.pdf
# https://arxiv.org/pdf/1504.04909.pdf - here we can find MAP-Elites description
# https://pdfs.semanticscholar.org/3461/99d0c8fce4bd91be855155bb1fe71844e9ac.pdf - polynomial mutation operator definition

import numpy as np
import random
import tensorflow as tf
from models.BaseModels.SequentialModel import SequentialModel


def map_elites(classifier, iter_max=10000, mutation_chance=0.5, mutation_power=15, min=0.0, max=1.0):
    """

    Args:
        classifier: A classifier model that we want to attack
        iter_max: Number of generations for wich we evolve our specimens
        mutation_chance: A change with wich mutation will occur for specimen feature
        mutation_power: Parameter controlling how far mutated feature can be from original value
        min: Minimal value of input image
        max: Maximal value of input image

    Returns:

    """
    return _map_elites(classifier,
                classifier.get_number_of_classes(),
                iter_max,
                mutation_chance,
                mutation_power,
                min,
                max)

def _map_elites(classifier, population_size, iter_max, mutation_chance=0.1, mutation_power=15, min=0.0, max=1.0,):
    """
    :type classifier: SequentialModel
    """
    performance_map = {}
    specimen_map = {}
    initial_population_size = 1000 * population_size

    for x in range(initial_population_size):
        new_specimen = np.random.uniform(min, max, classifier.get_input_shape()) #initialize specimens with random values
        eval_and_insert(performance_map, specimen_map, classifier, new_specimen)


    for i in range(iter_max):
        if i%100==0: print("generation: " + str(i))
        if i%2000==0: mutation_chance = mutation_chance * 0.5

        current_specimen = random.choice(list(specimen_map.values()))
        mutated_specimen = [mutate_specimen(current_specimen, min, max, mutation_chance, mutation_power=mutation_power) for i in range(2*population_size)]
        for specimen in mutated_specimen:
            eval_and_insert(performance_map, specimen_map, classifier, specimen)

    best_specimens = []
    best_logits = []
    for cur_class, specimen in specimen_map.items():
        specimen = tf.expand_dims(tf.convert_to_tensor(specimen, dtype=tf.float32), 0)
        best_specimens.append(specimen)
        logits = classifier(specimen)
        best_logits.append(logits)

    best_specimens = tf.concat(best_specimens, axis=0)
    best_logits = tf.concat(best_logits, axis=0)
    return (best_specimens, best_logits)


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

