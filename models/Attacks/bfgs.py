#This is the implementation of method given in this paper
#https://arxiv.org/pdf/1312.6199.pdf opis metody

import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.python.keras.losses import categorical_crossentropy
from models.BaseModels.SequentialModel import SequentialModel

def bfgs(data_sample, model, i_max, target_label, min=0.0, max=1.0):
    image, label = data_sample
    arr_image = []
    for i in range(len(image)):
        ret_image = _bfgs((tf.expand_dims(image[i], 0), label[i]), model, i_max, target_label, min, max)
        arr_image.append(ret_image)
    arr_image = tf.concat(arr_image, 0)
    return (arr_image, model(arr_image))

def _bfgs(data_sample, model, i_max, target_label, min=0.0, max=1.0):
    """

    :type model: SequentialModel
    """
    image, label = data_sample

    image = image.numpy().flatten()

    perturbation = [np.random.uniform(-image[i]/100, (max - image[i])/100) for i in range(len(image))]
    perturbation = np.array(perturbation)
    # perturbation = np.random.uniform(min, max/50, model.get_input_shape()).flatten().astype(dtype=np.float32)


    input_np_arr = image.reshape(model.get_input_shape()) + perturbation.reshape(model.get_input_shape())
    input_tensor = tf.convert_to_tensor(input_np_arr, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, 0)

    def min_func(perturbation, c, image):
        image = image.reshape(model.get_input_shape())
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        perturbation = perturbation.reshape(model.get_input_shape())
        perturbation = tf.convert_to_tensor(perturbation, dtype=tf.float32)



        with tf.GradientTape() as tape:
            tape.watch(perturbation)
            input_tensor = tf.add(image, perturbation)
            input_tensor = tf.expand_dims(input_tensor, 0)
            value = tf.add(tf.multiply(c, tf.linalg.norm(perturbation)),
                           categorical_crossentropy(target_label, model(input_tensor)))
        gradient = tape.gradient(value, perturbation)

        return value.numpy(), gradient.numpy().flatten().astype(np.float64)


    bounds = [(-image[i], max - image[i]) for i in range(len(image))]
    c = 1
    for i in range(10):
        new_perturbation, min_func_val, ret_dict = fmin_l_bfgs_b(min_func,
                                                                 perturbation,
                                                                 args=(c, image),
                                                                 maxiter=i_max,
                                                                 bounds=bounds)
        # print(ret_dict['grad'])
        input_np_arr = image.reshape(model.get_input_shape()) + new_perturbation.reshape(model.get_input_shape())
        input_tensor = tf.convert_to_tensor(input_np_arr, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, 0)
        if tf.squeeze(model(input_tensor)).numpy().argmax() == tf.argmax(tf.squeeze(target_label)).numpy():
            print("max c= " + str(c))
            break
        else:
            print("tested incerasing c= " + str(c))
            c = c*2

    c_high = c
    c_low = 0.0

    while c_high - c_low >= 1:
        c_half = (c_high + c_low)/2
        new_perturbation, min_func_val, ret_dict = fmin_l_bfgs_b(min_func,
                                                                 perturbation,
                                                                 args=(c_half, image),
                                                                 maxiter=i_max,
                                                                 bounds=bounds)
        # print(ret_dict['grad'])
        input_np_arr = image.reshape(model.get_input_shape()) + new_perturbation.reshape(model.get_input_shape())
        input_tensor = tf.convert_to_tensor(input_np_arr, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, 0)
        if tf.squeeze(model(input_tensor)).numpy().argmax() == tf.argmax(tf.squeeze(target_label)).numpy():
            print("tested halfing c= " + str(c_half))
            c_high = c_half
        else:
            print("tested halfing c= " + str(c_half))
            c_low = c_half




    input_np_arr = image.reshape(model.get_input_shape()) + new_perturbation.reshape(model.get_input_shape())
    input_tensor = tf.convert_to_tensor(input_np_arr, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, 0)
    return input_tensor

