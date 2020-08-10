#This is the implementation of method given in this paper
# https://arxiv.org/pdf/1805.11090.pdf
import tensorflow as tf
import tensorflow_probability as tfp

from liar_liar.attacks.attack import Attack
from liar_liar.models.base_models.sequential_model import SequentialModel


# TODO:Add updating parameters during runtime (mutation probability and mutation_range)

class GenAttack(Attack):

    def init_wrapper(self, *args, **kwargs):
        self._wrapper = gen_attack_wrapper(*args, **kwargs)

def gen_attack(classifier,
               data_sample,
               target_class,
               generation_nmb=100000,
               population_nmb=6,
               min=0.0,
               max=1.0,
               mutation_probability=0.05,
               delta=0.05):
    """
    Args:
        classifier (SequentialModel):
        data_sample (DatasetV2):
        target_class (Tensor): A one hot encoded label that specifies target class
        generation_nmb: Number of generations we want the algorithm to perform before returning a failed adversarial image
        population_nmb: Number of specimens we want to preserve from each generation
        min: Minimal value of input tensor
        max: Maximal value of input tensor
        mutation_probability: Probability of mutating a feature of specimen
        delta: maximal value by wich mutated value will change

    Returns:
        A tuple of generated adversarial sample and output of classifier for the adversarial sample

    """
    images, _ = data_sample
    ret_image = None
    for i in range(tf.shape(images)[0]):
        adv_image = _gen_attack(classifier,
                    images[i],
                    target_class,
                    generation_nmb,
                    population_nmb,
                    min,
                    max,
                    mutation_probability,
                    delta)
        ret_image = tf.expand_dims(adv_image, 0) if ret_image is None else tf.concat([ret_image, tf.expand_dims(adv_image, 0)], 0)

    parameters = {
        # "target_class": True,
        "generation_nmb": generation_nmb,
        "population_nmb": population_nmb,
        "mutation_probability": mutation_probability,
        "delta": delta,
        # "min": min,
        # "max": max
    }
    return (ret_image, classifier(tf.reshape(ret_image, tf.shape(images))), parameters)

def gen_attack_wrapper(generation_nmb=100000, population_nmb=6, min=0.0, max=1.0, mutation_probability=0.05, delta=0.05):
    """
    This wraps GenAttack call in a handy way that allows us using this as unspecified targeted attack method
    Returns: Wrapped GenAttack for targeted attack format

    """
    def wrapped_gen_attack(classifier, data_sample, target_class):
        return gen_attack(classifier,
                          data_sample,
                          target_class,
                          generation_nmb=generation_nmb,
                          population_nmb=population_nmb,
                          min=min,
                          max=max,
                          mutation_probability=mutation_probability,
                          delta=delta)
    return wrapped_gen_attack

@tf.function
def _gen_attack(classifier,
                image,
                target_class,
                generation_nmb=100000,
                population_nmb=6,
                min=0.0,
                max=1.0,
                mutation_probability=0.05,
                delta=0.05):
    """
    :type classifier: SequentialModel
    """
    target_class = tf.argmax(target_class)
    old_population = tf.tile(tf.expand_dims(image, 0), [population_nmb, 1,1,1])
    old_population = old_population + tf.random.uniform(old_population.shape, minval=-delta, maxval=delta)
    old_population = tf.map_fn(lambda x: clip_specimen(x, image, delta), old_population)
    new_population = tf.TensorArray(tf.float32, size=old_population.shape[0]) #this is just to satisfy the tf.function requirements

    for _ in tf.range(generation_nmb):
        fitness = tf.map_fn(lambda x: eval_specimen(target_class, x, classifier), old_population)
        highest_scoring_idx = tf.argmax(fitness)
        highest_scoring_idx = tf.reshape(highest_scoring_idx, []) #Make the Tensor into scalar tensor
        classification = tf.argmax(classifier(tf.expand_dims(old_population[highest_scoring_idx], 0)), axis=1)

        if tf.math.equal(classification, target_class):
            break

        new_population = new_population.write(0, tf.expand_dims(clip_specimen(old_population[highest_scoring_idx], image, delta),0))
        for i in tf.range(1, population_nmb):
            parents_idxs = pick_parents(fitness)
            first_parent_idx = parents_idxs[0]
            second_parent_idx = parents_idxs[1]
            child = crossover_parents(old_population[first_parent_idx], old_population[second_parent_idx], fitness[first_parent_idx], fitness[second_parent_idx])
            child = mutate_specimen(child, min, max, mutation_probability, delta)
            child = clip_specimen(child, image, delta)
            new_population = new_population.write(i, tf.expand_dims(child, 0))
        # Again reshape is just for tf.function sake
        old_population = tf.reshape(new_population.concat(), old_population.shape)

    fitness = tf.map_fn(lambda x: eval_specimen(target_class, x, classifier), old_population)
    highest_scoring_idx = tf.argmax(fitness)
    highest_scoring_idx = tf.reshape(highest_scoring_idx, [])  # Make the Tensor into scalar tensor
    # return tf.squeeze(old_population[highest_scoring_idx])
    return old_population[highest_scoring_idx]

# Clipping the specimen so that L-inf norm of org-adv isn't bigger than delta
@tf.function
def clip_specimen(child, image, delta):
    min = tf.clip_by_value(image - delta, 0.0, 1.0)
    max = tf.clip_by_value(image + delta, 0.0, 1.0)
    less = tf.math.less(child, min)
    greater = tf.math.greater(child, max)
    child = tf.where(less, min, child)
    return tf.where(greater, max, child)


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
    # return tf.clip_by_value(mutated, min, max)
    return mutated
