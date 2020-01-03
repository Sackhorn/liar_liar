import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def count(model, ret_image, labels):
    """

    :type labels: EagerTensor
    """
    labels = tf.dtypes.cast(labels, tf.float32)
    ret_labels = model(ret_image)
    ret_labels = tf.one_hot(tf.argmax(ret_labels, axis=1), model.get_number_of_classes())
    has_succeeded = tf.math.reduce_all(tf.math.equal(ret_labels, labels), axis=1)
    has_succeeded = list(has_succeeded.numpy())
    return has_succeeded.count(True)/len(has_succeeded)

def count_untargeted(model, ret_image, labels):
    ret_labels = model(ret_image)
    ret_labels = tf.one_hot(tf.argmax(ret_labels, axis=1), model.get_number_of_classes())
    has_succeeded = tf.math.reduce_all(tf.math.equal(ret_labels, labels), axis=1)
    has_succeeded = list(has_succeeded.numpy())
    return has_succeeded.count(False)/len(has_succeeded)

# This i straight from tensorboard tutorial https://www.tensorflow.org/tensorboard/image_summaries
def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image