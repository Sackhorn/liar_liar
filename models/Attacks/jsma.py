from mpl_toolkits import mplot3d
from tensorflow.python import enable_eager_execution, Tensor
from tensorflow.python.ops.losses.losses_impl import sparse_softmax_cross_entropy

from models.MNISTModels.ConvModel import ConvModel
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt



def show_plot(logits, image):
    """

    :type image: Tensor
    """
    probs = tf.nn.softmax(logits)
    probs = probs.numpy().reshape(10).tolist()
    fig = plt.figure()
    img_plt = fig.add_subplot(121)
    img_plt.imshow(image.numpy().reshape(28, 28).astype(np.float32), cmap=plt.get_cmap("gray"))
    # img_plt.imshow(tf.squeeze(image), cmap=plt.get_cmap("gray"))
    bar_plt = fig.add_subplot(122)
    bar_plt.bar(np.arange(10), probs)
    bar_plt.set_xticks(np.arange(10))
    plt.show()

def _fgsm(data_sample, model, i, eps, y_target=None, min=0.0, max=1.0):
    image, label = data_sample
    show_plot(model(image), image)
    for p in range(i):
        grds_by_cls = []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(image)
            logits = model(image)
            # gradient = tape.gradient(logits, image)
            for k in range(10):
                grds_by_cls.append(tf.squeeze(tape.gradient(logits[0][k], image)).numpy())
        saliency_map = np.zeros((10,28,28))
        for t in range(10):
            for x in range(28):
                for y in range(28):
                    sum_of_other = 0.0
                    for j in range(10):
                        sum_of_other += grds_by_cls[j][x][y] if j!=t else 0.0
                    if grds_by_cls[t][x][y] < 0.0 or sum_of_other > 0.0:
                        saliency_map[t][x][y] = 0.0
                    else:
                        saliency_map[t][x][y] = grds_by_cls[t][x][y]*abs(sum_of_other)
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # X, Y = np.meshgrid(np.arange(28), np.arange(28))
        # ax.plot_surface(X, Y, saliency_map[1])
        # plt.show()

        add_tensor = np.zeros((28,28))
        max_index = np.unravel_index(np.argmax(saliency_map[1]), saliency_map[1].shape)
        add_tensor[max_index] = 0.8
        add_tensor = add_tensor.reshape((1,28,28,1))
        add_tensor = tf.convert_to_tensor(add_tensor, dtype=tf.float32)
        # show_plot(logits, add_tensor)
        tf.reshape(add_tensor, (1,28,28,1))
        image = image + add_tensor
        image = tf.clip_by_value(image, min, max)
    show_plot(model(image), image)

def untargeted_fgsm(data_sample, model, i, eps, min=0.0, max=1.0):
    return _fgsm(data_sample, model, i, eps, min=min, max=max)

def targeted_fgsm(data_sample, model, i, eps, y_target, min=0.0, max=1.0):
    return _fgsm(data_sample, model, i, eps, y_target=y_target, min=min, max=max)


def test_fgsm_mnist():
    model = ConvModel()
    model.load_model_data()
    eval_dataset = model.get_dataset(tfds.Split.TEST, batch_size=1)
    target_label = tf.constant(7, dtype=tf.int64, shape=(1))
    for data_sample in eval_dataset.take(1):
        untargeted_fgsm(data_sample, model, 1000, 0.0001)

if __name__ == "__main__":
    enable_eager_execution()
    test_fgsm_mnist()
