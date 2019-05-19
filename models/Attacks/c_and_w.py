#https://arxiv.org/pdf/1312.6199.pdf opis metody + są jeszcze dwie kolejne publikacje tych samych autorów

from scipy.optimize import fmin_l_bfgs_b
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.python.ops.losses.losses_impl import sparse_softmax_cross_entropy
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow_datasets import Split

from models.CIFAR10Models.ConvModel import ConvModel
from models.MNISTModels.DenseModel import DenseModel
from models.utils.images import show_plot
from models.BaseModels.SequentialModel import SequentialModel

import tensorflow as tf
import numpy as np

def optimize_c_and_w(data_sample, model, i_max, target_label, c_val, min=0.0, max=1.0):
    """

    :type model: SequentialModel
    """

    image, label = data_sample
    perturbation = [np.random.uniform(-image[i], (max - image[i])) for i in range(len(image))]
    perturbation = np.array(perturbation)
    perturbation = perturbation.reshape(model.get_input_shape())
    perturbation = tf.convert_to_tensor(perturbation, dtype=tf.float32)
    perturbation = tf.Variable(perturbation)

    def loss():
        softmax = tf.squeeze(model(image+perturbation))
        highest_prob = tf.argmax(softmax)
        first = tf.concat([softmax[:highest_prob], softmax[highest_prob+1:]], 0) #get all elements except of the one with highest probability
        first = tf.reduce_max(first)
        second = softmax[target_label.numpy().squeeze().argmax()]
        g_func = tf.maximum(first - second, tf.constant(0, dtype=tf.float32))
        return tf.linalg.norm(perturbation) + c_val * g_func

    for i in range(i_max):
        AdamOptimizer().minimize(loss, var_list=[perturbation])
        # if i%100==0: show_plot(model(image + perturbation), image + perturbation, model.get_label_names())

    return perturbation

    # c_high = c
    # c_low = 0
    # for i in range(20):
    #     c_half = (c_high-c_low)/2
    #     # perturbation = init_perturbation()
    #     new_pert = optimize()
    #     if model(image + perturbation).numpy().squeeze().argmax() == target_label.numpy().squeeze().argmax():
    #         print("c_high = c_half tested halfing c= " + str(c_half))
    #         c_high = c_half
    #     else:
    #         print("c_low = c_half tested halfing c= " + str(c_half))
    #         c_low = c_half


def c_and_w(data_sample, model, i_max, target_label, min=0.0, max=1.0):
    image, label = data_sample
    c_high = 100
    c_low = 0
    c_half = None
    while c_high-c_low >= 1e-2:
        c_half = (c_high+c_low)/2
        perturbation = optimize_c_and_w(data_sample, model, i_max, target_label, c_half, min=min, max=max)

        if model(image + perturbation).numpy().squeeze().argmax() == target_label.numpy().squeeze().argmax():
            print("c_high = c_half tested halfing c= " + str(c_half))
            c_high = c_half
        else:
            print("c_low = c_half tested halfing c= " + str(c_half))
            c_low = c_half
        show_plot(model(image + perturbation), image + perturbation, model.get_label_names())

    print("c_half = " + str(c_half))
    perturbation = optimize_c_and_w(data_sample, model, i_max, target_label, c_half, min=min, max=max)
    show_plot(model(image + perturbation), image + perturbation, model.get_label_names())






def test_c_and_w():
    model = ConvModel()
    model.load_model_data()
    target_label = tf.one_hot(tf.constant(6, dtype=tf.int64, shape=(1)), model.get_number_of_classes())
    for data_sample in  model.get_dataset(Split.TEST, batch_size=1).take(1):
        c_and_w(data_sample, model, 10000, target_label)

if __name__ == "__main__":
    enable_eager_execution()
    for i in range(1):
        test_c_and_w()
