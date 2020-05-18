#This is the implementation of method given in this paper
#https://arxiv.org/pdf/1312.6199.pdf opis metody

import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.python.keras.losses import categorical_crossentropy
from liar_liar.base_models.sequential_model import SequentialModel



def bfgs(classifier, data_sample, target_class, iter_max, min=0.0, max=1.0):
    image, label = data_sample
    arr_image = []
    for i in range(len(image)):
        ret_image = _bfgs(classifier, (tf.expand_dims(image[i], 0), label[i]), target_class, iter_max, min, max)
        arr_image.append(ret_image)
    arr_image = tf.concat(arr_image, 0)
    parameteres = {
        "target_class":int(tf.argmax(target_class.numpy())),
        "iter_max":iter_max,
        "min":min,
        "max":max
    }
    return (arr_image, classifier(arr_image), parameteres)

def bfgs_wrapper(iter_max=100, min=0.0, max=1.0):
    """
    This wraps bfgs call in a handy way that allows us using this as unspecified targeted attack method

    Returns: Wrapped BFGS for targeted attack format

    """
    def wrapped_bfgs(classifier, data_sample, target_class):
        return bfgs(classifier, data_sample, target_class, iter_max=iter_max, min=min, max=max)
    return wrapped_bfgs

def _bfgs(classifier, data_sample, target_class, iter_max, min=0.0, max=1.0):
    """

    :type classifier: SequentialModel
    """

    def min_func(perturbation, c, image):
        image = image.reshape(classifier.get_input_shape())
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        perturbation = perturbation.reshape(classifier.get_input_shape())
        perturbation = tf.convert_to_tensor(perturbation, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(perturbation)
            input_tensor = tf.add(image, perturbation)
            input_tensor = tf.expand_dims(input_tensor, 0)
            value = tf.add(tf.multiply(c, tf.linalg.norm(perturbation)),
                           categorical_crossentropy(target_class, classifier(input_tensor)))
        gradient = tape.gradient(value, perturbation)
        return value.numpy(), gradient.numpy().flatten().astype(np.float64)

    target_class = tf.expand_dims(target_class, 0)
    image, label = data_sample
    image = image.numpy().flatten()
    perturbation = [np.random.uniform(-image[i]/100, (max - image[i])/100) for i in range(len(image))]
    perturbation = np.array(perturbation)
    bounds = [(-image[i], max - image[i]) for i in range(len(image))]

    c = 1
    for i in range(10):
        new_perturbation, min_func_val, ret_dict = fmin_l_bfgs_b(min_func,
                                                                 perturbation,
                                                                 args=(c, image),
                                                                 maxiter=iter_max,
                                                                 bounds=bounds)
        input_tensor = numpy_to_tensor(classifier, image, new_perturbation)
        if is_in_target_class(classifier, input_tensor, target_class):
            break
        c = c*2

    c_high = c
    c_low = 0.0
    while c_high - c_low >= 1:
        c_half = (c_high + c_low)/2
        new_perturbation, min_func_val, ret_dict = fmin_l_bfgs_b(min_func,
                                                                 perturbation,
                                                                 args=(c_half, image),
                                                                 maxiter=iter_max,
                                                                 bounds=bounds)
        input_tensor = numpy_to_tensor(classifier, image, new_perturbation)
        if is_in_target_class(classifier, input_tensor, target_class):
            c_high = c_half
        else:
            c_low = c_half

    input_tensor = numpy_to_tensor(classifier, image, new_perturbation)
    return input_tensor


def numpy_to_tensor(classifier, image, new_perturbation):
    input_np_arr = image.reshape(classifier.get_input_shape()) + new_perturbation.reshape(classifier.get_input_shape())
    input_tensor = tf.convert_to_tensor(input_np_arr, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, 0)
    return input_tensor

def is_in_target_class(classifier, input_tensor, target_class):
    return tf.squeeze(classifier(input_tensor)).numpy().argmax() == tf.argmax(tf.squeeze(target_class)).numpy()

