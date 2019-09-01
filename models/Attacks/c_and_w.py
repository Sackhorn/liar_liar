#https://arxiv.org/pdf/1312.6199.pdf opis metody + są jeszcze dwie kolejne publikacje tych samych autorów

import time
import tensorflow as tf
import numpy as np

from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow_datasets import Split

from models.CIFAR10Models.ConvModel import ConvModel
from models.MNISTModels.DenseModel import DenseModel
from models.utils.images import show_plot
from models.BaseModels.SequentialModel import SequentialModel


@tf.function
def optimize_c_and_w(image, model, i_max, target_label, c_val, perturbation, optimizer):
    """

    :type model: SequentialModel
    """

    target_label_index = tf.argmax(target_label, output_type=tf.int32, axis=1)
    target_label_index = tf.reshape(target_label_index, [])
    for i in tf.range(i_max):
        with tf.GradientTape() as tape:
            tape.watch(perturbation)
            delta = 0.5 * (tf.tanh(perturbation) + 1)
            softmax = tf.squeeze(model(delta, get_raw=True))
            first = tf.concat([softmax[:, :target_label_index], softmax[:, target_label_index + 1:]], 1)  # get all elements except of the one with highest probability
            first = tf.reduce_max(first, axis=1)
            second = softmax[:, target_label_index]
            g_func = first - second
            loss =  tf.square(tf.linalg.norm(delta - image)) + c_val * g_func
        gradient = tape.gradient(loss, perturbation)
        optimizer.apply_gradients([(gradient, perturbation)])
    return 0.5 * (tf.tanh(perturbation) + 1)

def c_and_w(data_sample, model, i_max, target_label, min=0.0, max=1.0):
    image, label = data_sample
    show_plot(model(image), image, model.get_label_names())
    c_high = 100
    c_low = 0
    c_half = None
    for i in range(20):
        c_half = (c_high+c_low)/2

        start = time.time()

        perturbation = tf.Variable(np.random.normal(min, max, image.shape), dtype=tf.float32)
        optimizer = GradientDescentOptimizer(1e-2)
        new_image = optimize_c_and_w(image, model, i_max, target_label, tf.constant(c_half, dtype=tf.float32), perturbation, optimizer)

        end = time.time()
        print("one calculation took: " + str(end-start))

        if model(new_image).numpy().squeeze().argmax() == target_label.numpy().squeeze().argmax():
            print("c_high = c_half tested halfing c= " + str(c_half))
            c_high = c_half
            if c_high - c_low <= 0.01: break
        else:
            print("c_low = c_half tested halfing c= " + str(c_half))
            c_low = c_half
        show_plot(model(new_image), new_image, model.get_label_names())

    print("c_half = " + str(c_half))
    perturbation = tf.Variable(np.random.normal(min, max, image.shape), dtype=tf.float32)
    optimizer = GradientDescentOptimizer(1e-2)
    new_image = optimize_c_and_w(image, model, i_max, target_label, tf.constant(c_half, dtype=tf.float32), perturbation, optimizer)
    show_plot(model(new_image), new_image, model.get_label_names())




def test_c_and_w():
    # model = DenseModel().load_model_data()
    model = ConvModel().load_model_data()
    target_label = tf.one_hot(tf.constant(1, dtype=tf.int64, shape=(1)), model.get_number_of_classes())
    batch_size = 10
    for data_sample in  model.get_dataset(Split.TEST, batch_size=batch_size, shuffle=1).take(1):
        image, label = data_sample
        perturbation = tf.Variable(np.random.normal(0.0, 1.0, image.shape), dtype=tf.float32)
        optimizer = GradientDescentOptimizer(1e-2)

        start = time.time()
        ret_image = optimize_c_and_w(image, model, 10000, target_label, tf.constant(1, dtype=tf.float32), perturbation, optimizer)

        end = time.time()
        print("calculations took: " + str(end - start))
        for i in range(batch_size):
            show_plot(model(tf.expand_dims(image[i], axis=0)), tf.expand_dims(image[i], axis=0), model.get_label_names())
            show_plot(model(tf.expand_dims(ret_image[i], axis=0)), tf.expand_dims(ret_image[i], axis=0), model.get_label_names())

if __name__ == "__main__":

    test_c_and_w()
