#https://arxiv.org/pdf/1312.6199.pdf opis metody + są jeszcze dwie kolejne publikacje tych samych autorów

from scipy.optimize import fmin_l_bfgs_b
from tensorflow.python import enable_eager_execution
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow_datasets import Split

from models.MNISTModels.ConvModel import ConvModel
from models.MNISTModels.DenseModel import DenseModel
from models.MNISTModels.LeNet5 import LeNet5
from models.utils.images import show_plot
from models.BaseModels.SequentialModel import SequentialModel

import tensorflow as tf
import numpy as np

def optimize_c_and_w(data_sample, model, i_max, target_label, c_val, min=0.0, max=1.0):
    """

    :type model: SequentialModel
    """

    image, label = data_sample

    perturbation = tf.Variable(np.random.normal(min, max, image.shape), dtype=tf.float32)

    def loss():
        delta = 0.5 * (tf.tanh(perturbation) + 1)
        softmax = tf.squeeze(model(delta))
        target_label_index = target_label.numpy().squeeze().argmax()
        first = tf.concat([softmax[:target_label_index], softmax[target_label_index+1:]], 0) #get all elements except of the one with highest probability
        first = tf.reduce_max(first)
        second = softmax[target_label_index]
        g_func = first - second
        return tf.square(tf.linalg.norm(delta-image)) + c_val * g_func

    for i in range(i_max):
        AdamOptimizer().minimize(loss, var_list=[perturbation])
        # if i%500==0:
        #     delta = (0.5 * (tf.tanh(perturbation) + 1))
        #     show_plot(model(delta), delta, model.get_label_names())

    return 0.5 * (tf.tanh(perturbation) + 1)

def c_and_w(data_sample, model, i_max, target_label, min=0.0, max=1.0):
    image, label = data_sample
    show_plot(model(image), image, model.get_label_names())
    c_high = 1000
    c_low = 0
    c_half = None
    for i in range(20):
        c_half = (c_high+c_low)/2
        new_image = optimize_c_and_w(data_sample, model, i_max, target_label, c_half, min=min, max=max)

        if model(new_image).numpy().squeeze().argmax() == target_label.numpy().squeeze().argmax():
            print("c_high = c_half tested halfing c= " + str(c_half))
            c_high = c_half
            if c_high - c_low <= 0.01: break
        else:
            print("c_low = c_half tested halfing c= " + str(c_half))
            c_low = c_half
        show_plot(model(new_image), new_image, model.get_label_names())

    print("c_half = " + str(c_half))
    new_image = optimize_c_and_w(data_sample, model, 10 * i_max, target_label, c_half, min=min, max=max)
    show_plot(model(new_image), new_image, model.get_label_names())

def test_c_and_w():
    model = ConvModel()
    model.load_model_data()
    target_label = tf.one_hot(tf.constant(3, dtype=tf.int64, shape=(1)), model.get_number_of_classes())
    for data_sample in  model.get_dataset(Split.TEST, batch_size=1).take(1):
        c_and_w(data_sample, model, 1000, target_label)
        # optimize_c_and_w(data_sample, model, 10000, target_label, 10)

if __name__ == "__main__":
    enable_eager_execution()
    for i in range(1):
        test_c_and_w()
