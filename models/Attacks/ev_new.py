import random
import numpy as np
import tensorflow as tf
from tensorflow.python import enable_eager_execution
from tensorflow_datasets import Split

from models.CIFAR10Models.ConvModel import ConvModel
from models.ImageNet.InceptionV3Wrapper import ResNetWrapper
from models.MNISTModels.DenseModel import DenseModel
from models.MNISTModels.LeNet5 import LeNet5
from models.utils.images import show_plot

# TODO: Generalize for all models
# https://arxiv.org/pdf/1805.11090.pdf Strategia ewolucyjna
from models.utils.utils import count


def evolutionary_attack(model, data_sample, target_class, generation_nmb=1000, population_nmb=10, min=0.0, max=1.0, mutation_chance=0.3):
    """
    :type model: SequentialModel
    """

    image, label = data_sample
    np_image = image.numpy()
    shape = np_image.shape
    old_population = [(np_image + np.random.uniform(-0.05, 0.05, shape)).astype(np.float32) for _ in range(population_nmb)] #creating the initial population
    for k in range(generation_nmb):

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
            # print("gen: " + str(k) + " fitness: " + str(np.array(fitness).max()))
            fitness = tf.nn.softmax(fitness)
            fitness = fitness.numpy()

        for _ in range(len(old_population)-1):
            first_parent_idx, second_parent_idx = pick_parents(fitness)
            child = crossover_parents(old_population[first_parent_idx], old_population[second_parent_idx], fitness[first_parent_idx], fitness[second_parent_idx])
            child = mutate_specimen(child, min, max, mutation_chance)
            child = clip_child(child, image)
            new_population.append(child)

        old_population = new_population

    # print("run out of generation numbers")
    fitness = list(map(lambda x: eval_specimen(target_class, x, model, get_raw=True), old_population))
    highest_scoring_idx = np.argmax(fitness)
    output = old_population[highest_scoring_idx].reshape(model.get_input_shape())
    output = tf.convert_to_tensor(output, dtype=tf.float32)
    output = tf.expand_dims(output, 0)
    return output

def clip_child(child, image):
    min = image - 0.1
    max = image + 0.1
    return tf.clip_by_value(child, min, max).numpy()

def pick_parents(fitness):

    first = np.random.choice(np.arange(len(fitness)), p=fitness)
    second = np.random.choice(np.arange(len(fitness)), p=fitness)
    return (first, second)


def crossover_parents(first, second, first_fitness, second_fitness):
    prob_on_pick_first = first_fitness / (first_fitness + second_fitness)
    first_copy = np.copy(first)
    second_copy = np.copy(second)
    with np.nditer([first_copy, second_copy], op_flags=['readwrite']) as it:
        for first_val, second_val in it:
            if np.random.uniform() > prob_on_pick_first:
                first_val[...] = second_val
    return first_copy

def eval_specimen(target_class, specimen, model, get_raw):
    # TODO Getting raw should not return logits but fitness function for class
    input = specimen.reshape(model.get_input_shape())
    input = tf.convert_to_tensor(input, dtype=tf.float32)
    input = tf.expand_dims(input, 0)
    logits = model(input, get_raw=get_raw)
    logits = tf.squeeze(logits).numpy()
    probs_without_target = np.array([logits[i] for i in range(len(logits)) if i!=target_class ])
    probs_without_target = probs_without_target[np.nonzero(probs_without_target)] # We prune elements that are so close to zero that it doesnt matter
    fitness = np.log(logits[target_class]) - np.log(np.sum(probs_without_target))
    # fitness = np.log(logits[target_class])
    return fitness

def get_classification(specimen, model):
    input = specimen.reshape(model.get_input_shape())
    input = tf.convert_to_tensor(input, dtype=tf.float32)
    input = tf.expand_dims(input, 0)
    logits = model(input)
    return np.argmax(tf.squeeze(logits).numpy())

def mutate_specimen(specimen, min, max, mutation_chance):
    new_specimen = np.copy(specimen)
    delta = 0.05
    with np.nditer(new_specimen, op_flags=['readwrite']) as it:
        for old_val in it:
            if np.random.uniform() <= mutation_chance:
                new_val = old_val + random.uniform(-delta, delta)
                old_val[...] = new_val
    return np.clip(new_specimen, min, max)

def test_ea_attack():
    # model = LeNet5().load_model_data()
    model = ConvModel().load_model_data()
    # model = ResNetWrapper().load_model_data()
    count_arr = []
    target_class = 9
    for data_sample in model.get_dataset(Split.TEST, batch_size=1, augment_data=False):
        image, label = data_sample
        # model(image)
        # show_plot(model(image), image, labels_names=model.get_label_names())
        ret_image = evolutionary_attack(model, data_sample, target_class=target_class, mutation_chance=0.05)
        # show_plot(model(ret_image), ret_image, labels_names=model.get_label_names())
        if tf.argmax(model(ret_image), axis=1) == target_class:
            count_ret = 1.0
        else:
            count_ret = 0.0
        print(count_ret)
        count_arr.append(count_ret)
        avg = np.average(np.array(count_arr))
        print("average: " + str(avg))

if __name__ == "__main__":
    enable_eager_execution()
    test_ea_attack()
