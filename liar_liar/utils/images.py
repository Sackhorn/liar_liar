import json
from os import path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import Tensor

# imagenet_labels_path = path.dirname(path.realpath(__file__))
# imagenet_labels_path = path.join(imagenet_labels_path, path.pardir, path.pardir, "imagenet1000_clsidx_to_labels.txt")
# imagenet_labels_path = imagenet_labels_path.join([path.pardir, path.pardir, "imagenet1000_clsidx_to_labels.txt"])
# imagenet_labels_dict = None
# with open(imagenet_labels_path, 'r', encoding='utf-8') as f:
#     imagenet_labels_dict = json.load(f)
# print(imagenet_labels_dict)


def show_plot(logits, image, labels_names=None, plot_title=None):
    """

    :type logits: Tensor
    :type image: Tensor
    """
    top_k_display = labels_names is not None and len(logits.numpy().squeeze()) > 15
    labels_names = np.arange(logits.numpy().size) if labels_names is None else np.array(labels_names)
    probs = logits.numpy().flatten()
    if top_k_display:
        sorted_probs_idx = np.argsort(probs)[-10:]
        sorted_probs = probs.take(sorted_probs_idx)
        probs = sorted_probs
        labels_names = labels_names.take(sorted_probs_idx)
    probs = probs.tolist()
    labels_names = labels_names.tolist()



    plt.subplot(1, 2, 1)
    plt.imshow(tf.squeeze(image), cmap=plt.get_cmap("gray"))

    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(probs)), probs)
    plt.xticks(np.arange(len(probs)), labels_names, rotation=90)
    if plot_title is not None:
        plt.title(plot_title)
    plt.show()