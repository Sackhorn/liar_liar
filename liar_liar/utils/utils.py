import io
import logging
import os

import tensorflow as tf
import matplotlib.pyplot as plt


#TODO: this could be improve not sure if correct
def batch_image_norm(image):
    return tf.norm(tf.norm(tf.norm(image, axis=3), axis=2), axis=1)

# This is straight from tensorboard tutorial https://www.tensorflow.org/tensorboard/image_summaries
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

def disable_logging():
    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0, alsologtostdout=False)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False