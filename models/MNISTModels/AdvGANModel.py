import time

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import categorical_crossentropy, BinaryCrossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow_core.python.keras.layers import BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, \
    Dropout, Flatten
from tensorflow_datasets import Split

from models.MNISTModels.MNISTModel import MNISTModel
from models.utils.images import show_plot


class MNISTGenerator(MNISTModel):

    def __init__(self, optimizer=Adam(1e-4), loss=categorical_crossentropy, metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="mnist_generator")
        self.sequential_layers = [
        Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
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
generator.trainable = True
discriminator.trainable = True
# output = generator(tf.random.uniform([1, 100]))
# logits = discriminator(output)
# show_plot(logits, output)

EPOCHS = 200
BATCH_SIZE = 256
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])
checkpoint = tf.train.Checkpoint(generator_optimizer=generator.optimizer,
                                 discriminator_optimizer=discriminator.optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    generator.trainable = True
    discriminator.trainable = True
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, train=True)

      real_output = discriminator(images, train=True)
      fake_output = discriminator(generated_images, train=True)

      gen_loss = generator.generator_loss(fake_output)
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
        # image = image_batch
        train_step(image)

    # Produce images for the GIF as we go
    # display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)
    #
    # Save the model every 15 epochs
    # if (epoch + 1) % 15 == 0:
    #   checkpoint.save(file_prefix = generator.SAVE_DIR)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  # display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  generator.trainable = False
  discriminator.trainable = False
  predictions = model(test_input, train=False)
  # show_plot(tf.random.uniform([1,10]),predictions[0])

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')

  # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


dataset = generator.get_dataset(Split.TRAIN, batch_size=256, augment_data=False)
# checkpoint.restore(tf.train.latest_checkpoint(generator.SAVE_DIR))
# generator.load_model_data()
# discriminator.load_model_data()
train(dataset, EPOCHS)
generator.save_model_data()
discriminator.save_model_data()
#
#
# generator.load_model_data()
# discriminator.load_model_data()
# for i in range(10):
#     seed = tf.random.normal([num_examples_to_generate, noise_dim])
#     generate_and_save_images(generator, 0, seed)
