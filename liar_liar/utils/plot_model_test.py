import tensorflow as tf
from tensorflow.python import keras

# os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz 2.43.20200408.0903'
from liar_liar.models.mnist_models.mnist_dense_model import MNISTDenseModel

model = MNISTDenseModel()
input = keras.layers.Input((28,28,1))
first_layer = model.sequential_layers[0]
output = first_layer(input)
for layer in model.sequential_layers[1:]:
    output = layer(output)

new_model = keras.Model(inputs=input, outputs=output)
new_model.summary()

tf.keras.utils.plot_model(new_model, to_file='model.jpg', show_shapes=True, rankdir='TB')

