import random
import numpy as np
import tensorflow as tf
from tensorflow.python import enable_eager_execution
from tensorflow_datasets import Split

from models.CIFAR10Models.ConvModel import ConvModel
from models.MNISTModels.DenseModel import DenseModel
from models.MNISTModels.LeNet5 import LeNet5
from models.utils.images import show_plot

# TODO: Generalize for all models
# https://arxiv.org/pdf/1805.11090.pdf Strategia ewolucyjna



def evolutionary_attack(model, data_sample, target_class, generation_nmb=500, population_nmb=100, min=0.0, max=1.0, mutation_chance=0.3):
    """
    :type model: SequentialModel
    """

    image, label = data_sample
    old_population = [mutate_specimen(image.numpy(), min, max, mutation_chance) for i in range(population_nmb)] #creating the initial population

    for k in range(generation_nmb):
        print(k)
        fitness = list(map(lambda x: eval_specimen(target_class, x, model, get_raw=False), old_population))
        highest_scoring_idx = np.argmax(fitness)
        classification = get_classification(old_population[highest_scoring_idx], model)

        if classification == target_class:
            output = old_population[highest_scoring_idx].reshape(model.get_input_shape())
            output = tf.convert_to_tensor(output, dtype=tf.float32)
            output = tf.expand_dims(output, 0)
            return output
        else:
            new_population = [old_population[highest_scoring_idx]]
            fitness = list(map(lambda x: eval_specimen(target_class, x, model, get_raw=False), old_population))

        for _ in range(len(old_population)-1):
            first_parent_idx, second_parent_idx = pick_parents(fitness)
            child = crossover_parents(old_population[first_parent_idx], old_population[second_parent_idx], fitness[first_parent_idx], fitness[second_parent_idx])
            child = mutate_specimen(child, min, max, mutation_chance)
            new_population.append(child)

        old_population = new_population

    print("run out of generation numbers")
    fitness = list(map(lambda x: eval_specimen(target_class, x, model, get_raw=True), old_population))
    highest_scoring_idx = np.argmax(fitness)
    output = old_population[highest_scoring_idx].reshape(model.get_input_shape())
    output = tf.convert_to_tensor(output, dtype=tf.float32)
    output = tf.expand_dims(output, 0)
    return output

def pick_parents(fitness):
    parents_list = [None, None]
    tmp = np.array(fitness)
    normalized_fitness = tmp/tmp.max()
    for j in range(2):
        while parents_list[j] is None:
            probability = np.random.uniform()
            for i, softmax_prob in enumerate(normalized_fitness):
                if probability < softmax_prob:
                    parents_list[j] = i
                    break
    return tuple(parents_list)


def crossover_parents(first, second, first_fitness, second_fitness):
    prob_on_pick_first = first_fitness/ (first_fitness+second_fitness)
    first_copy = np.copy(first)
    second_copy = np.copy(second)
    with np.nditer([first_copy, second_copy], op_flags=['readwrite']) as it:
        for first_val, second_val in it:
            if not np.random.uniform() < prob_on_pick_first:
                first_val[...] = second_val
    return first_copy

def eval_specimen(target_class, specimen, model, get_raw):
    # TODO Getting raw should not return logits but fitness function for class
    input = specimen.reshape(model.get_input_shape())
    input = tf.convert_to_tensor(input, dtype=tf.float32)
    input = tf.expand_dims(input, 0)
    logits = model(input, get_raw=get_raw)
    return tf.squeeze(logits).numpy()[target_class]

def get_classification(specimen, model):
    input = specimen.reshape(model.get_input_shape())
    input = tf.convert_to_tensor(input, dtype=tf.float32)
    input = tf.expand_dims(input, 0)
    logits = model(input)
    return np.argmax(tf.squeeze(logits).numpy())

def mutate_specimen(specimen, min, max, mutation_chance):
    new_specimen = np.copy(specimen)
    delta = 0.1
    with np.nditer(new_specimen, op_flags=['readwrite']) as it:
        for old_val in it:
            if np.random.uniform() <= mutation_chance:
                new_val = old_val + random.uniform(-delta, delta)
                old_val[...] = new_val
    return np.clip(new_specimen, min, max)

def test_ea_attack():
    model = LeNet5().load_model_data()
    model = ConvModel().load_model_data()
    for data_sample in model.get_dataset(Split.TEST, batch_size=1, augment_data=False).take(10):
        image, label = data_sample
        # model(image)
        show_plot(model(image), image, labels_names=model.get_label_names())
        ret_image = evolutionary_attack(model, data_sample, target_class=1, mutation_chance=0.1)
        show_plot(model(ret_image), ret_image, labels_names=model.get_label_names())

if __name__ == "__main__":
    enable_eager_execution()
    test_ea_attack()
