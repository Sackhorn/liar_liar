import time

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import categorical_crossentropy, BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow_core.python.keras.layers import BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, \
    Dropout, Flatten
from tensorflow_datasets import Split

from models.BaseModels.SequentialModel import SequentialModel
from models.MNISTModels.ConvModel import ConvModel
from models.MNISTModels.MNISTModel import MNISTModel
import numpy as np
from models.utils.images import show_plot
from models.utils.utils import count


class MNISTGenerator(MNISTModel):

    def __init__(self, optimizer=Adam(1e-4), loss=categorical_crossentropy, metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="mnist_generator")
        self.sequential_layers = [
        Flatten(),
        Dense(7 * 7 * 256, use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Reshape((7, 7, 256)),
        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'),
        ]
        self.gan_loss = BinaryCrossentropy(from_logits=True)

    def call(self, input, get_raw=False, train=False, **kwargs):
        output = super().call(input, get_raw=get_raw)
        return output + input

    def generator_loss(self, fake_output):
        return self.gan_loss(tf.ones_like(fake_output), fake_output)


class MNISTDiscriminator(MNISTModel):

    def __init__(self, optimizer=Adam(1e-4), loss=categorical_crossentropy, metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="mnist_discriminator")
        self.sequential_layers = [
            Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 1]),
            LeakyReLU(),
            Dropout(0.3),
            Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            LeakyReLU(),
            Dropout(0.3),
            Flatten(),
            Dense(1),
        ]
        self.gan_loss = BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.gan_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.gan_loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

generator = MNISTGenerator()
discriminator = MNISTDiscriminator()


EPOCHS = 250
BATCH_SIZE = 250
noise_dim = 100
num_examples_to_generate = 16


classifier: SequentialModel = ConvModel().load_model_data()
classifier.trainable = False
seed = None
for data in classifier.get_dataset(Split.TRAIN, batch_size=4).take(1):
    seed, _ = data
classifier_loss = CategoricalCrossentropy(from_logits=True)
target_class = tf.one_hot(tf.constant(1, shape=[BATCH_SIZE], dtype=tf.int32), depth=10, dtype=tf.int32)
generator.trainable = True
discriminator.trainable = True
alpha = tf.constant(1.0, dtype=tf.float32)
beta = tf.constant(1.0, dtype=tf.float32)
c_constant = tf.constant(0.0, dtype=tf.float32)

# @tf.function
def train_step(images):
    generator.trainable = True
    discriminator.trainable = True
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

      generated_images = generator(images, train=True)
      real_output = discriminator(images, train=True)
      fake_output = discriminator(generated_images, train=True)
      classifier_output = classifier(generated_images, train=False)

      gen_loss = generator.generator_loss(fake_output) + alpha * classifier_loss(target_class, classifier_output) + beta * tf.math.maximum(0.0, tf.norm(generated_images-images, axis=1)-c_constant)
      disc_loss = discriminator.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator.optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
        image, label = image_batch
        train_step(image)
    generate_and_save_images(generator, epoch + 1, seed, dataset)
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

def generate_and_save_images(model, epoch, test_input, dataset):
  generator.trainable = False
  discriminator.trainable = False
  predictions = model(seed)
  predictions = generator(seed)
  probs = classifier(predictions, train=False)

  for i in range(predictions.shape[0]):
      plt.subplot(2,4, 2*(i+1)-1)
      plt.imshow(tf.clip_by_value(predictions[i, :, :, 0], 0.0, 1.0), cmap='gray')
      plt.xticks([])
      plt.yticks([])
      plt.subplot(2,4,2*(i+1))
      plt.bar(np.arange(len(probs[i])), probs[i])
      plt.xticks(np.arange(len(probs[i])), np.arange(probs[i].numpy().size), rotation=90)
      plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  accuracy = []
  diff_norm = []
  for image, label in dataset:
      generated = model(image)
      diff_norm.append(tf.math.reduce_mean(tf.norm(generated - image, axis=1), axis=0).numpy())
      accuracy.append(count(classifier, generated, target_class))
  print("accuracy: " + str(np.average(np.array(accuracy))), " mean noise norm: " + str(np.average(np.array(diff_norm))))

def train_gan():
    dataset = generator.get_dataset(Split.TRAIN, batch_size=BATCH_SIZE, augment_data=False)
    # generator.load_model_data()
    # discriminator.load_model_data()
    train(dataset, EPOCHS)
    generator.save_model_data()
    discriminator.save_model_data()

def test_model():
    generator.load_model_data()
    discriminator.load_model_data()
    dataset = generator.get_dataset(Split.TRAIN, batch_size=4, augment_data=False).take(10)
    for data in dataset:
        image, _ = data
        generate_and_save_images(generator, 0, image, dataset)

if __name__=="__main__":
    train_gan()
    # test_model()