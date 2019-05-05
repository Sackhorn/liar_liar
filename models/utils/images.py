import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def show_plot(logits, image):
    """

    :type image: Tensor
    """
    probs = logits
    probs = probs.numpy().flatten().tolist()
    fig = plt.figure()
    img_plt = fig.add_subplot(121)
    img_plt.imshow(tf.squeeze(image), cmap=plt.get_cmap("gray"))
    bar_plt = fig.add_subplot(122)
    bar_plt.bar(np.arange(len(probs)), probs)
    bar_plt.set_xticks(np.arange(len(probs)))
    plt.show()