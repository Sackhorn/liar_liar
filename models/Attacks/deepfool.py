from tensorflow.python import enable_eager_execution, Tensor
from models.MNISTModels.ConvModel import ConvModel
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


# TODO: Generalize for all models
# Source https://arxiv.org/pdf/1511.07528.pdf
from models.MNISTModels.DenseModel import DenseModel


def show_plot(logits, image):
    """

    :type image: Tensor
    """
    # probs = tf.nn.softmax(logits)
    probs = logits
    probs = probs.numpy().reshape(10).tolist()
    fig = plt.figure()
    img_plt = fig.add_subplot(121)
    img_plt.imshow(image.numpy().reshape(28, 28).astype(np.float32), cmap=plt.get_cmap("gray"))
    bar_plt = fig.add_subplot(122)
    bar_plt.bar(np.arange(10), probs)
    bar_plt.set_xticks(np.arange(10))
    plt.show()

def deepfool(data_sample, model, min=0.0, max=1.0):
    image, label = data_sample
    label = tf.argmax(tf.squeeze(label)).numpy()
    show_plot(model(image), image)
    iter = 0
    while tf.argmax(tf.squeeze(model(image))).numpy() == label:

        logits = None
        gradients_by_cls = []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(image)
            logits = model(image)
            for k in range(10):
                gradients_by_cls.append(tf.squeeze(tape.gradient(logits[0][k], image)).numpy())

        logits = tf.squeeze(logits).numpy()
        w_prime = []
        f_prime = []
        for k in range(10):
            if k == tf.squeeze(label).numpy():
                f_prime.append(float('+inf'))
                w_prime.append(float("+inf"))
                continue
            w_prime.append(gradients_by_cls[k] - gradients_by_cls[label])
            f_prime.append(logits[k] - logits[label])
        tmp = []
        for k in range(10):
            if k == tf.squeeze(label).numpy():
                tmp.append(float("+inf"))
                continue
            tmp.append(abs(f_prime[k]) / np.linalg.norm(w_prime[k], ord=2))
        l = np.argmin(tmp)
        perturbation = (abs(f_prime[l]) * w_prime[l]) / np.square(np.linalg.norm(w_prime[l], ord=2))
        image = image + tf.reshape(tf.convert_to_tensor(perturbation, dtype=tf.float32), (1, 28, 28, 1))
        image = tf.clip_by_value(image, min, max)
        # show_plot(model(image), image)

    show_plot(model(image), image)


def test_deepfool():
    model = DenseModel()
    model.load_model_data()
    eval_dataset = model.get_dataset(tfds.Split.TEST, batch_size=1)
    target_label = 5
    for data_sample in eval_dataset.take(1):
        deepfool(data_sample, model)

if __name__ == "__main__":
    enable_eager_execution()
    test_deepfool()
