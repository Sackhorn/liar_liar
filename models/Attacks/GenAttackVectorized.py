import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from models.Attacks.attack import Attack

# https://arxiv.org/pdf/1805.11090.pdf Strategia ewolucyjna
# TODO:Add updating parameters during runtime (mutation probability and mutation_range)
# for vectorization tf.tensor_scatter_nd_add
# for vectorization tfp.distributions.Bernoulli

@tf.function
def evolutionary_attack(model,
                        image,
                        target_class,
                        generation_nmb=100000,
                        population_nmb=6,
                        min=0.0,
                        max=1.0,
                        mutation_probability=0.05,
                        delta=0.05):
    """
    :type model: SequentialModel
    """
    target_class = tf.argmax(target_class)
    old_population = tf.tile(tf.expand_dims(image, 0), [population_nmb, 1,1,1])
    old_population = old_population + tf.random.uniform(old_population.shape, minval=-delta, maxval=delta)
    new_population = tf.TensorArray(tf.float32, size=old_population.shape[0]) #this is just to satisfy the tf.function requirements

    for _ in tf.range(generation_nmb):
        fitness = tf.map_fn(lambda x: eval_specimen(target_class, x, model), old_population)
        highest_scoring_idx = tf.argmax(fitness)
        highest_scoring_idx = tf.reshape(highest_scoring_idx, []) #Make the Tensor into scalar tensor
        classification = tf.argmax(model(tf.expand_dims(old_population[highest_scoring_idx], 0)), axis=1)

        if tf.math.equal(classification, target_class):
            break

        new_population = new_population.write(0, tf.expand_dims(old_population[highest_scoring_idx],0))
        for i in tf.range(1, population_nmb-1):
            parents_idxs = pick_parents(fitness)
            first_parent_idx = parents_idxs[0]
            second_parent_idx = parents_idxs[1]
            child = crossover_parents(old_population[first_parent_idx], old_population[second_parent_idx], fitness[first_parent_idx], fitness[second_parent_idx])
            child = mutate_specimen(child, min, max, mutation_probability, delta)
            child = clip_specimen(child, image, delta)
            new_population = new_population.write(i, tf.expand_dims(child, 0))
        # Again reshape is just for tf.function sake
        old_population = tf.reshape(new_population.concat(), old_population.shape)

    fitness = tf.map_fn(lambda x: eval_specimen(target_class, x, model), old_population)
    highest_scoring_idx = tf.argmax(fitness)
    highest_scoring_idx = tf.reshape(highest_scoring_idx, [])  # Make the Tensor into scalar tensor
    return tf.squeeze(old_population[highest_scoring_idx])

# Clipping the specimen so that L-inf norm of org-adv isn't bigger than delta
@tf.function
def clip_specimen(child, image, delta):
    min = image - delta
    max = image + delta
    return tf.clip_by_value(child, min, max)

# pick two random specimens according to distribution density function given by softmax of fitness function
@tf.function
def pick_parents(fitness):
    new_fitness = tf.nn.softmax(tf.reshape(fitness, [-1]))
    return tfp.distributions.Categorical(probs=new_fitness).sample(2)

@tf.function
def crossover_parents(first, second, first_fitness, second_fitness):
    prob_on_pick_first = first_fitness / (first_fitness + second_fitness)
    shape = tf.shape(first)
    child = tf.where(
        tf.random.uniform(shape) > prob_on_pick_first,
        second,
        first
    )
    return child

# Calculate fitness function for given specimen
@tf.function
def eval_specimen(target_class, specimen, model):
    specimen = tf.expand_dims(specimen, 0)
    logits = model(specimen)
    probs_without_target = tf.concat([logits[:, target_class:], logits[:, target_class+1:]], axis=1)
    # We prune elements that are so close to zero that it doesnt matter and that may cause log to explode to infinity
    probs_without_target = tf.boolean_mask(probs_without_target, probs_without_target>0)
    #calculating fitness function
    fitness = tf.math.log(logits[:, target_class]) - tf.math.log(tf.math.reduce_sum(probs_without_target))
    return fitness

@tf.function
def mutate_specimen(specimen, min, max, mutation_probability, delta):
    shape = tf.shape(specimen)
    mutated = tf.where(
        tf.random.uniform(shape) <= mutation_probability,
        specimen + tf.random.uniform(shape, minval=-delta, maxval=delta),
        specimen
    )
    return tf.clip_by_value(mutated, min, max)


class GenAttackVectorized(Attack):
    @staticmethod
    def run_attack(model, data_sample, target_class):
        images, _ = data_sample
        ret_image = tf.map_fn(lambda x: evolutionary_attack(model, x, target_class), images)
        return (ret_image, model(ret_image))
