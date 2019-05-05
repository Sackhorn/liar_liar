from tensorflow.python import enable_eager_execution
from models.utils.images import show_plot
from models.MNISTModels.DenseModel import DenseModel
from models.BaseModels.SequentialModel import SequentialModel

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# TODO: Generalize for all models
# Source https://arxiv.org/pdf/1511.07528.pdf


def deepfool(data_sample, model, max_iter=100, min=0.0, max=1.0):
    """

    :type model: SequentialModel
    """

    nmb_classes = model.get_number_of_classes()
    image, label = data_sample
    label = tf.argmax(tf.squeeze(label)).numpy()
    show_plot(model(image), image)
    iter = 0
    while tf.argmax(tf.squeeze(model(image))).numpy() == label and iter < max_iter:
        logits = None
        gradients_by_cls = []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(image)
            logits = model(image, get_raw=True)
            for k in range(nmb_classes):
                gradients_by_cls.append(tf.squeeze(tape.gradient(logits[0][k], image)).numpy())

        logits = tf.squeeze(logits).numpy()
        w_prime = []
        f_prime = []
        for k in range(nmb_classes):
            if k == tf.squeeze(label).numpy():
                f_prime.append(float('+inf'))
                w_prime.append(float("+inf"))
                continue
            w_prime.append(gradients_by_cls[k] - gradients_by_cls[label])
            f_prime.append(logits[k] - logits[label])
        tmp = []
        for k in range(nmb_classes):
            if k == tf.squeeze(label).numpy():
                tmp.append(float("+inf"))
                continue
            tmp.append(abs(f_prime[k]) / np.linalg.norm(w_prime[k]))
        l = np.argmin(tmp)
        perturbation = (abs(f_prime[l]) * w_prime[l]) / np.square(np.linalg.norm(w_prime[l]))
        image = image + tf.reshape(tf.convert_to_tensor(perturbation, dtype=tf.float32), model.get_input_shape())
        image = tf.clip_by_value(image, min, max)
        iter += 1
    show_plot(model(image), image)


def test_deepfool():
    model = DenseModel()
    model.load_model_data()
    eval_dataset = model.get_dataset(tfds.Split.TEST, batch_size=1)
    target_label = 5
    for data_sample in eval_dataset.take(10):
        deepfool(data_sample, model)

if __name__ == "__main__":
    enable_eager_execution()
    test_deepfool()
