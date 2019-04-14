import tensorflow as tf
from MNISTModel import *
import numpy as np


def show_plot(logits, image):
    probs = tf.nn.softmax(logits)
    probs = probs.numpy().reshape(10).tolist()
    fig = plt.figure()
    img_plt = fig.add_subplot(121)
    img_plt.imshow(image.numpy().reshape(28, 28).astype(np.float32), cmap=plt.get_cmap("gray"))
    bar_plt = fig.add_subplot(122)
    bar_plt.bar(np.arange(10), probs)
    bar_plt.set_xticks(np.arange(10))
    plt.show()

def fgsm(x, y_target, model, i, eps, min=0.0, max=1.0):
    show_plot(model(x), x)

    for i in range(i):
        with tf.GradientTape() as tape:
            tape.watch(x)
            logits = model(x)
            loss = tf.losses.sparse_softmax_cross_entropy(y_target, logits)
        gradient = tape.gradient(loss, x)
        x = x - eps * tf.sign(gradient)
        x = tf.clip_by_value(x, min, max)

    show_plot(model(x), x)
    return x

def test_fgsm_mnist():
    model = MNISTModel()
    model.load_model_data()
    eval_dataset = model.get_dataset(tfds.Split.TEST, batch_size=1)
    target_label = tf.constant(7, dtype=tf.int64, shape=(1))
    for i in eval_dataset.take(1):
        image, label = i['image'], i['label']
        fgsm(image, target_label, model, 100, 0.001)

if __name__ == "__main__":
    tf.enable_eager_execution()
    test_fgsm_mnist()
