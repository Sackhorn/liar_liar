import tensorflow as tf

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from liar_liar.models.base_models.AdvGAN import AdvGAN
from liar_liar.models.mnist_models import MNISTConvModel
from liar_liar.models.mnist_models.mnist_model_base import MNISTModel

class MNISTGenerator(MNISTModel):

    def __init__(self, optimizer=Adam(learning_rate=0.0001), loss=None, metrics=[None]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="mnist_generator")
        self.sequential_layers = [
        Conv2D(16, (5,5), strides=(2,2), padding='same'),
        BatchNormalization(),
        LeakyReLU(),
        Conv2D(32, (5, 5), strides=(2,2), padding='same'),
        BatchNormalization(),
        LeakyReLU(),
        Conv2D(64, (5, 5), strides=(1,1),  padding='same'),
        BatchNormalization(),
        LeakyReLU(),
        Conv2D(128, (5, 5), strides=(1, 1), padding='same'),
        BatchNormalization(),
        LeakyReLU(),
        Conv2D(256, (5, 5), strides=(1, 1), padding='same'),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2DTranspose(64, (5,5), strides=(1,1), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2DTranspose(16, (5, 5), strides=(2,2), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2DTranspose(1, (4, 4),  padding='same', activation='tanh'),
        ]

        # self.gan_loss = MeanSquaredError()
        self.gan_loss = BinaryCrossentropy()
        self.loss = self.generator_loss

    def call(self, input, get_raw=False, train=False, **kwargs):
        output = super().call(input, get_raw=get_raw)
        return output

    def generator_loss(self, fake_output):
        return self.gan_loss(tf.ones_like(fake_output), fake_output)


class MNISTDiscriminator(MNISTModel):

    def __init__(self, optimizer=Adam(learning_rate=0.0003), loss=None, metrics=[None]):
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
        # self.gan_loss = MeanSquaredError()
        self.gan_loss = BinaryCrossentropy()
        self.loss = self.discriminator_loss

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.gan_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.gan_loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


GAN = AdvGAN(MNISTGenerator(), MNISTDiscriminator(), MNISTConvModel())
GAN.gan_train(alpha=1.0, beta=1.0, c_constant=0.3, target_class=5, batch_size=256, epochs=200)