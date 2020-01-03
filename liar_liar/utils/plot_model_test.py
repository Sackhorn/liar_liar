import tensorflow
from tensorflow.python import keras

from liar_liar.mnist_models.mnist_conv_model import MNISTConvModel
from liar_liar.mnist_models.mnist_dense_model import MNISTDenseModel
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


model = MNISTDenseModel()
# TODO: implement model.get_shape() method
# input = keras.layers.Input(model.get_shape())
input = keras.layers.Input((28,28,1))
first_layer = model.sequential_layers[0]
output = first_layer(input)
for layer in model.sequential_layers[1:]:
    output = layer(output)

new_model = keras.Model(inputs=input, outputs=output)
new_model.summary()

keras.utils.plot_model(new_model, to_file='model.png', show_shapes=True, rankdir='TB')

