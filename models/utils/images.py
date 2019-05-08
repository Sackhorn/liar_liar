import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import Tensor
def show_plot(logits, image, labels_names=None):
    """

    :type logits: Tensor
    :type image: Tensor
    """
    labels_names = np.arange(logits.numpy().size) if labels_names is None else labels_names

    probs = logits.numpy().flatten().tolist()

    plt.subplot(1, 2, 1)
    plt.imshow(tf.squeeze(image), cmap=plt.get_cmap("gray"))

    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(probs)), probs)
    plt.xticks(np.arange(len(probs)), labels_names, rotation=90)

    plt.show()