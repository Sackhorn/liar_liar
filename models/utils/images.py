import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.BaseModels.SequentialModel import SequentialModel

def show_plot(logits, image, labels_names):
    """

    :type model: Tensor
    :type image: Tensor
    """
    probs = logits.numpy().flatten().tolist()

    plt.subplot(1, 2, 1)
    plt.imshow(tf.squeeze(image), cmap=plt.get_cmap("gray"))

    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(probs)), probs)
    plt.xticks(np.arange(len(probs)), labels_names, rotation=90)

    plt.show()