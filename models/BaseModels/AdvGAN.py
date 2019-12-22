import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow_datasets import Split
from models.BaseModels.DataProvider import DataProvider
from models.BaseModels.SequentialModel import SequentialModel
from models.utils.utils import count

DISPLAY_NUMBER = 6
class AdvGAN(DataProvider):
    def __init__(self, generator_model, discriminator_model, classifier):
        """

        :type classifier: SequentialModel
        :type discriminator_model: SequentialModel
        :type generator_model: SequentialModel
        """
        super().__init__()
        self.generator = generator_model
        self.discriminator = discriminator_model
        self.classifier = classifier.load_model_data()
        self.register_data_provider(classifier.MODEL_NAME, classifier.dataset_name, classifier.data_dir)

    @tf.function
    def call(self, input, get_raw=False, train=False, **kwargs):
        return self.generator(input)

    def gan_train(self, alpha, beta, c_constant, target_class, batch_size, epochs):
        """
        :type batch_size: int
        :type target_class: int
        """
        self.alpha = tf.constant(alpha)
        self.beta = tf.constant(beta)
        self.c_constant = tf.constant(c_constant)
        self.target_class = tf.constant(target_class, shape=[batch_size], dtype=tf.int32)
        self.target_class = tf.one_hot(self.target_class, depth=self.get_number_of_classes(), dtype=tf.int32)
        self.generator.trainable = True
        self.discriminator.trainable = True
        self.classifier.trainable = False
        display_input_dataset = self.get_dataset(Split.TRAIN, batch_size=DISPLAY_NUMBER).take(1)
        display_input, _ = tf.data.experimental.get_single_element(display_input_dataset)
        test_dataset = self.get_dataset(Split.TEST, batch_size=batch_size, augment_data=False)
        train_dataset = self.get_dataset(Split.TRAIN, batch_size=batch_size, augment_data=False)
        for epoch in range(epochs):
            start = time.time()
            for image_batch in train_dataset:
                image, _ = image_batch
                self.train_step(image)
            self.accuracy_and_diff(test_dataset)
            self.plot_sample_output(display_input, epoch + 1)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


    @tf.function
    def train_step(self, images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(images)
            real_output = self.discriminator(images)
            fake_output = self.discriminator(generated_images)
            classifier_output = self.classifier(generated_images)
            gen_loss = self.generator.generator_loss(fake_output)
            adv_loss = self.alpha * self.classifier.loss(self.target_class, classifier_output)
            hinge_distance_loss =  self.beta * tf.math.maximum(0.0, tf.norm(generated_images - images, axis=1) - self.c_constant)
            gen_total_loss = gen_loss + adv_loss + hinge_distance_loss
            disc_loss = self.discriminator.discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def plot_sample_output(self, display_input, epoch):
        generated_images = self.generator(display_input)
        probs = self.classifier(generated_images)
        figure = plt.figure(figsize=(10, 10))
        for i in range(generated_images.shape[0]):
            plt.subplot(3, 4, 2 * (i + 1) - 1)
            plt.imshow(tf.clip_by_value(generated_images[i, :, :, 0], 0.0, 1.0), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 4, 2 * (i + 1))
            plt.bar(np.arange(len(probs[i])), probs[i])
            plt.xticks(np.arange(len(probs[i])), np.arange(probs[i].numpy().size))
        plt.show()
        return figure

    def accuracy_and_diff(self, test_dataset):
        accuracy = []
        diff_norm = []
        for image, label in test_dataset:
            generated = self.generator(image)
            diff_norm.append(tf.math.reduce_mean(tf.norm(generated - image, axis=1), axis=0).numpy())
            accuracy.append(count(self.classifier, generated, self.target_class))
        avg_accuracy = np.average(np.array(accuracy))
        avg_diff_norm = np.average(np.array(diff_norm))
        print("accuracy: {} mean noise norm: {}".format(avg_accuracy, avg_diff_norm))