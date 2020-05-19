import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import Tensor


#Show just one image with it's logits
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

#Compare original image and adversary version
def show_plot_comparison(adv_image,
                         adv_logits,
                         orig_image,
                         orig_logits,
                         labels_names,
                         plot_title=None,
                         target_class=None,
                         true_class=None):
    """

    :type orig_logits: Tensor
    :type orig_image: Tensor
    """
    labels_names = np.arange(orig_logits.numpy().size) if labels_names is None else np.array(labels_names)

    orig_logits = orig_logits.numpy().flatten()
    orig_colors = ['blue'] * len(orig_logits)
    adv_logits = adv_logits.numpy().flatten()
    adv_colors = ['blue']*len(adv_logits)

    true_class = int(tf.argmax(true_class).numpy())
    orig_colors[true_class], adv_colors[true_class] = 'green', 'green'

    if target_class is not None:
        target_class = int(tf.argmax(target_class).numpy()) if target_class is not None else None
        orig_colors[target_class], adv_colors[target_class] = 'red', 'red'
    orig_zip = list(zip(orig_logits, labels_names, orig_colors))
    adv_zip = list(zip(adv_logits, labels_names, adv_colors))
    #case when we have more than 10 classes in dataset
    if len(orig_logits) > 10:
        orig_zip = sorted(orig_zip, key=lambda x: x[0], reverse=True)
        adv_zip = sorted(adv_zip, key=lambda x: x[0], reverse=True)
        orig_zip = orig_zip[:10]
        adv_zip = adv_zip[:10]
    orig_logits, orig_labels, orig_colors = list(zip(*orig_zip))
    adv_logits, adv_labels, adv_colors = list(zip(*adv_zip))

    figure = plt.figure(figsize=(6,9))
    if plot_title is not None:
        figure.suptitle(plot_title)

    plt.subplot(2, 2, 1)
    plt.imshow(tf.squeeze(orig_image), cmap=plt.get_cmap("gray"))
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.bar(np.arange(len(orig_logits)), orig_logits, color=orig_colors)
    plt.xticks(np.arange(len(orig_logits)), orig_labels, rotation=90)
    plt.axis('tight')

    plt.subplot(2, 2, 3)
    plt.imshow(tf.squeeze(adv_image), cmap=plt.get_cmap("gray"))
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.bar(np.arange(len(adv_logits)), adv_logits, color=adv_colors)
    plt.xticks(np.arange(len(adv_logits)), adv_labels, rotation=90)
    plt.axis('tight')

    plt.show()
