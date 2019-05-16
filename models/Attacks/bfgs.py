#https://arxiv.org/pdf/1312.6199.pdf opis metody

from scipy.optimize import fmin_l_bfgs_b
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.python.ops.losses.losses_impl import sparse_softmax_cross_entropy
from tensorflow_datasets import Split

from models.CIFAR10Models.ConvModel import ConvModel
from models.MNISTModels.DenseModel import DenseModel
from models.utils.images import show_plot
from models.BaseModels.SequentialModel import SequentialModel

import tensorflow as tf
import numpy as np

def bfgs(data_sample, model, i_max, target_label, min=0.0, max=1.0):
    """

    :type model: SequentialModel
    """
    image, label = data_sample

    image = image.numpy().flatten()
    perturbation = np.random.uniform(min, max, model.get_input_shape()).flatten().astype(dtype=np.float32)

    def min_func(perturbation, c, image):
        input_np_arr = image.reshape(model.get_input_shape()) + perturbation.reshape(model.get_input_shape())
        input_tensor = tf.convert_to_tensor(input_np_arr, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, 0)
        return c * np.linalg.norm(perturbation) + sparse_softmax_cross_entropy(target_label, model(input_tensor))

    c = 0.0001
    while True:
        new_perturbation, min_func_val, ret_dict = fmin_l_bfgs_b(min_func,
                                                                 perturbation,
                                                                 args=(c, image),
                                                                 approx_grad=True,
                                                                 maxiter=i_max,
                                                                 epsilon=0.01)
        # print(ret_dict['grad'])
        input_np_arr = image.reshape(model.get_input_shape()) + new_perturbation.reshape(model.get_input_shape())
        input_tensor = tf.convert_to_tensor(input_np_arr, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, 0)
        show_plot(model(input_tensor), input_np_arr, model.get_label_names())
        if tf.squeeze(model(input_tensor)).numpy().argmax() == tf.squeeze(target_label).numpy():
            print("max c= " + str(c))
            break
        else:
            print("tested incerasing c= " + str(c))
            c = 2*c

    c_high = c
    c_low = 0.0

    while c_high - c_low > 0.0001:
        c_half = (c_high + c_low)/2
        new_perturbation, min_func_val, ret_dict = fmin_l_bfgs_b(min_func,
                                                                 perturbation,
                                                                 args=(c_half, image),
                                                                 approx_grad=True,
                                                                 maxiter=i_max,
                                                                 epsilon=0.01)
        # print(ret_dict['grad'])
        input_np_arr = image.reshape(model.get_input_shape()) + new_perturbation.reshape(model.get_input_shape())
        input_tensor = tf.convert_to_tensor(input_np_arr, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, 0)
        show_plot(model(input_tensor), input_np_arr, model.get_label_names())
        if tf.squeeze(model(input_tensor)).numpy().argmax() == tf.squeeze(target_label).numpy():
            print("tested halfing c= " + str(c_half))
            c_high = c_half
        else:
            print("tested halfing c= " + str(c_half))
            c_low = c_half




    input_np_arr = image.reshape(model.get_input_shape()) + new_perturbation.reshape(model.get_input_shape())
    input_tensor = tf.convert_to_tensor(input_np_arr, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, 0)


def test_fgsm_mnist():
    model = ConvModel()
    model.load_model_data()
    target_label = tf.constant(6, dtype=tf.int64, shape=(1))
    for data_sample in  model.get_dataset(Split.TEST, batch_size=1).take(1):
        bfgs(data_sample, model, 10, target_label)

if __name__ == "__main__":
    enable_eager_execution()
    for i in range(3):
        test_fgsm_mnist()
