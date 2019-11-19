#https://arxiv.org/pdf/1312.6199.pdf opis metody + są jeszcze dwie kolejne publikacje tych samych autorów
import tensorflow as tf
import numpy as np
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from models.Attacks.attack import Attack

@tf.function
def optimize_c_and_w(image, model, i_max, target_label, c_val, perturbation, optimizer):
    target_label_index = tf.argmax(target_label, output_type=tf.int32, axis=0)
    target_label_index = tf.reshape(target_label_index, [])
    tf.print(c_val)

    for _ in tf.range(i_max):
        with tf.GradientTape() as tape:
            tape.watch(perturbation)
            delta = 0.5 * (tf.tanh(perturbation) + 1)
            logits = tf.squeeze(model(delta, get_raw=True))
            first = tf.concat([logits[:, :target_label_index], logits[:, target_label_index + 1:]], 1)  # get all elements except of the one with highest probability
            first = tf.reduce_max(first, axis=1)
            second = logits[:, target_label_index]
            g_func = first - second
            loss =  tf.square(tf.linalg.norm(delta - image)) + tf.multiply(c_val, g_func)
        gradient = tape.gradient(loss, perturbation)
        optimizer.apply_gradients([(gradient, perturbation)])

    return 0.5 * (tf.tanh(perturbation) + 1)

@tf.function
def c_and_w(image, model, target_label, perturbation, optimizer, iter_max=10000, binary_iter_max=10):
    c_high = tf.fill([image.shape[0]], 100.0)
    c_low = tf.fill([image.shape[0]], 0.01)
    prev_high = c_high

    for _ in tf.range(binary_iter_max):
        c_half = (c_high + c_low) / 2
        new_image = optimize_c_and_w(image, model, iter_max, target_label, c_half, perturbation, optimizer)
        output = model(new_image)
        output = tf.one_hot(tf.argmax(output, axis=1), model.get_number_of_classes())
        has_succeed = tf.math.reduce_all(tf.math.equal(output, target_label), axis=1)
        prev_high = c_high
        c_high = tf.where(has_succeed, c_half, c_high)
        c_low = tf.where(has_succeed, c_low, c_half)
        tf.compat.v1.assign(perturbation, tf.random.uniform(perturbation.shape))

    return optimize_c_and_w(image, model, iter_max, target_label, prev_high, perturbation, optimizer)

class CarliniWagner(Attack):
    @staticmethod
    def run_attack(model, data_sample, target_class):
        image, label = data_sample
        perturbation = tf.Variable(np.random.uniform(0.0, 1.0, image.shape), dtype=tf.float32)
        optimizer = GradientDescentOptimizer(1e-2)
        target_label = tf.one_hot(target_class, model.get_number_of_classes())
        iter_max = 10000
        return_image = c_and_w(image, model, target_label, perturbation, optimizer, iter_max)
        return (return_image, model(image))
