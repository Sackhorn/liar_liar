#This is the implementation of method given in this paper
# https://arxiv.org/abs/1608.04644
import tensorflow as tf
import numpy as np
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

from liar_liar.utils.utils import batch_image_norm


def carlini_wagner(classifier,
                   data_sample,
                   target_class,
                   optimizer=GradientDescentOptimizer(1e-2),
                   optimization_iter=10000,
                   binary_iter=20,
                   c_high=1000.0,
                   c_low=0.0,
                   kappa=0.0):
    """

    Args:
        classifier: A classifier model that we want to attack
        data_sample (): A tuple of tensors of structure (image_tensor, label_tensor) against wich attack is run
        target_class: One hot encoded target class for the attack
        optimizer: Optimizer used in algorithm
        optimization_iter: Number of steps take in optimization for given parameter
        binary_iter: Number of steps take in binary search for parameter C
        c_high: Upper limit of range used in binary search for parameter C
        c_low: lower limit of range used in binary search for parameter C

    Returns: A tuple of structure (adversarial_example, classifier output for examples)

    """
    image, label = data_sample
    perturbation = tf.Variable(np.random.uniform(0.0, 1.0, image.shape), dtype=tf.float32)
    cw = tf.function(_c_and_w)
    # cw = _c_and_w
    return_image = cw(image, classifier, target_class, perturbation, optimizer, optimization_iter, binary_iter, c_high, c_low, kappa)
    parameters = {
        "optimizer":optimizer,
        "optimization_iter":optimization_iter,
        "binary_iter":binary_iter,
        "C_high":c_high,
        "c_low":c_low,
        "kappa":kappa}
    return (return_image, classifier(return_image), parameters)

def carlini_wagner_wrapper(optimizer=GradientDescentOptimizer(1e-2),
                           optimization_iter=10000,
                           binary_iter=20,
                           c_high=1000.0,
                           c_low=0.0,
                           kappa=0.0):
    """
    This wraps Carlini and Wagner call in a handy way that allows us using this as unspecified targeted attack method
    Returns: Wrapped Carlini and Wagner for targeted attack format

    """
    def wrapped_carlini_wagner(classifier, data_sample, target_class):
        return carlini_wagner(classifier, data_sample, target_class, optimizer, optimization_iter, binary_iter, c_high, c_low, kappa)
    return wrapped_carlini_wagner


def _optimize_c_and_w(image, classifier, iter_max, target_class, c_val, perturbation, optimizer, kappa):
    target_label_index = tf.argmax(target_class, output_type=tf.int32, axis=0)
    target_label_index = tf.reshape(target_label_index, [])
    tf.print(c_val)

    for _ in tf.range(iter_max):
        with tf.GradientTape() as tape:
            tape.watch(perturbation)
            delta = 0.5 * (tf.tanh(perturbation) + 1)
            logits = tf.squeeze(classifier(delta, get_raw=True))
            # get all elements except of the one with highest probability
            first = tf.concat([logits[:, :target_label_index], logits[:, target_label_index + 1:]], 1)#TODO: if batch=1 then this doesn't work
            first = tf.reduce_max(first, axis=1)
            second = logits[:, target_label_index]
            g_func = tf.maximum(first - second, -kappa)
            loss =  tf.square(batch_image_norm(delta - image)) + tf.multiply(c_val, g_func)
        gradient = tape.gradient(loss, perturbation)
        optimizer.apply_gradients([(gradient, perturbation)])

    return 0.5 * (tf.tanh(perturbation) + 1)

def _c_and_w(image, model, target_class, perturbation, optimizer, optimization_iter, binary_iter, c_high, c_low, kappa):
    c_high = tf.fill([image.shape[0]], c_high)
    c_low = tf.fill([image.shape[0]], c_low)
    prev_high = c_high

    for _ in tf.range(binary_iter):
        c_half = (c_high + c_low) / 2
        new_image = _optimize_c_and_w(image, model, optimization_iter, target_class, c_half, perturbation, optimizer, kappa)
        output = model(new_image)
        output = tf.one_hot(tf.argmax(output, axis=1), model.get_number_of_classes())
        has_succeed = tf.math.reduce_all(tf.math.equal(output, target_class), axis=1)
        prev_high = c_high
        c_high = tf.where(has_succeed, c_half, c_high)
        c_low = tf.where(has_succeed, c_low, c_half)
        tf.compat.v1.assign(perturbation, tf.random.uniform(perturbation.shape))

    return _optimize_c_and_w(image, model, optimization_iter, target_class, prev_high, perturbation, optimizer, kappa)
