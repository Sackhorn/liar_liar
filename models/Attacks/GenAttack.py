import random
import numpy as np
import tensorflow as tf
from tensorflow.python import enable_eager_execution
from tensorflow_datasets import Split

from models.Attacks.attack import Attack
from models.CIFAR10Models.ConvModel import ConvModel
from models.ImageNet.InceptionV3Wrapper import ResNetWrapper
from models.MNISTModels.DenseModel import DenseModel
from models.MNISTModels.LeNet5 import LeNet5
from models.utils.images import show_plot

# https://arxiv.org/pdf/1805.11090.pdf Strategia ewolucyjna
# TODO:Add updating parameters during runtime (mutation probability and mutation_range)
# for vectorization tf.tensor_scatter_nd_add
# for vectorization tfp.distributions.Bernoulli


def evolutionary_attack(model,
                        image,
                        target_class,
                        generation_nmb=1000,
                        population_nmb=10,
                        min=0.0,
                        max=1.0,
                        mutation_probability=0.05,
                        delta=0.05):
    """
    :type model: SequentialModel
    """

    np_image = image.numpy()
    shape = np_image.shape
    old_population = [(np_image + np.random.uniform(-delta, delta, shape)).astype(np.float32) for _ in range(population_nmb)] #creating the initial population
    for k in range(generation_nmb):

        fitness = list(map(lambda x: eval_specimen(target_class, x, model), old_population))
        highest_scoring_idx = np.argmax(fitness)
        classification = get_classification(old_population[highest_scoring_idx], model)

        if classification == target_class:
            output = old_population[highest_scoring_idx].reshape(model.get_input_shape())
            output = tf.convert_to_tensor(output, dtype=tf.float32)
            output = tf.expand_dims(output, 0)
            return output
        else:
            new_population = [old_population[highest_scoring_idx]]
            fitness = list(map(lambda x: eval_specimen(target_class, x, model), old_population))
            fitness = tf.nn.softmax(fitness)
            fitness = fitness.numpy()

        for _ in range(len(old_population)-1):
            first_parent_idx, second_parent_idx = pick_parents(fitness)
            child = crossover_parents(old_population[first_parent_idx], old_population[second_parent_idx], fitness[first_parent_idx], fitness[second_parent_idx])
            child = mutate_specimen(child, min, max, mutation_probability, delta)
            child = clip_child(child, image, delta)
            new_population.append(child)

        old_population = new_population

    fitness = list(map(lambda x: eval_specimen(target_class, x, model), old_population))
    highest_scoring_idx = np.argmax(fitness)
    output = old_population[highest_scoring_idx].reshape(model.get_input_shape())
    output = tf.convert_to_tensor(output, dtype=tf.float32)
    output = tf.expand_dims(output, 0)
    return output

# Clipping the specimen so that L-inf norm of org-adv isn't bigger than delta
def clip_child(child, image, delta):
    min = image - delta
    max = image + delta
    return tf.clip_by_value(child, min, max).numpy()

# pick two random specimens according to distribution density function given by softmax of fitness function
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

# Calculate fitness function for given specimen
def eval_specimen(target_class, specimen, model):
    input = specimen.reshape(model.get_input_shape())
    input = tf.convert_to_tensor(input, dtype=tf.float32)
    input = tf.expand_dims(input, 0)
    logits = model(input)
    logits = tf.squeeze(logits).numpy()
    probs_without_target = np.array([logits[i] for i in range(len(logits)) if i!=target_class ])
    # We prune elements that are so close to zero that it doesnt matter and that may cause log to explode to infinity
    probs_without_target = probs_without_target[np.nonzero(probs_without_target)]
    #calculating fitness function
    fitness = np.log(logits[target_class]) - np.log(np.sum(probs_without_target))
    return fitness

def get_classification(specimen, model):
    input = specimen.reshape(model.get_input_shape())
    input = tf.convert_to_tensor(input, dtype=tf.float32)
    input = tf.expand_dims(input, 0)
    logits = model(input)
    return np.argmax(tf.squeeze(logits).numpy())

def mutate_specimen(specimen, min, max, mutation_probability, delta):
    new_specimen = np.copy(specimen)
    with np.nditer(new_specimen, op_flags=['readwrite']) as it:
        for old_val in it:
            if np.random.uniform() <= mutation_probability:
                new_val = old_val + random.uniform(-delta, delta)
                old_val[...] = new_val
    return np.clip(new_specimen, min, max)

class GenAttack(Attack):
    @staticmethod
    def run_attack(model, data_sample, target_class):
        target_class = tf.argmax(target_class).numpy()
        images, _ = data_sample
        image_array = tf.TensorArray(tf.float32, size=images.shape[0])
        # TODO: check in case it isn't batched
        for idx in range(images.shape[0]):
            ret_image = evolutionary_attack(model, images[idx], target_class)
            image_array.write(idx, ret_image)
        ret_image = tf.squeeze(image_array.stack())
        return (ret_image, model(ret_image))
