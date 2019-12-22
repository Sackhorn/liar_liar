import tensorflow as tf

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from models.BaseModels.AdvGAN import AdvGAN
from models.MNISTModels.ConvModel import ConvModel
from models.MNISTModels.MNISTModel import MNISTModel

class MNISTGenerator(MNISTModel):

    def __init__(self, optimizer=Adam(1e-4), loss=None, metrics=[None]):
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
        Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        ]

        self.gan_loss = BinaryCrossentropy(from_logits=True)
        self.loss = self.generator_loss

    def call(self, input, get_raw=False, train=False, **kwargs):
        output = super().call(input, get_raw=get_raw)
        return output + input

    def generator_loss(self, fake_output):
        return self.gan_loss(tf.ones_like(fake_output), fake_output)


class MNISTDiscriminator(MNISTModel):

    def __init__(self, optimizer=Adam(1e-4), loss=None, metrics=[None]):
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
        self.loss = self.discriminator_loss

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.gan_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.gan_loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

GAN = AdvGAN(MNISTGenerator(), MNISTDiscriminator(), ConvModel())
GAN.gan_train(alpha=10.0, beta=1.0, c_constant=0.0, target_class=1, batch_size=250, epochs=100)