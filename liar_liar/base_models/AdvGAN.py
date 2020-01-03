import datetime
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import wandb

from tensorflow_datasets import Split
from liar_liar.base_models.data_provider import DataProvider
from liar_liar.base_models.sequential_model import SequentialModel
from liar_liar.utils.utils import count, plot_to_image

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
        self.register_data_provider(classifier.MODEL_NAME+"_gan", classifier.dataset_name, classifier.data_dir)

    @tf.function
    def call(self, input, get_raw=False, train=False, **kwargs):
        return self.generator(input) + input

    def gan_train(self, alpha, beta, c_constant, target_class, batch_size, epochs):
        """
        :type batch_size: int
        :type target_class: int
        """
        self.initialize_metrics()
        self.alpha = tf.constant(alpha)
        self.beta = tf.constant(beta)
        self.c_constant = tf.constant(c_constant)
        self.target_class = tf.constant(target_class, shape=[batch_size], dtype=tf.int32)
        self.target_class = tf.one_hot(self.target_class, depth=self.get_number_of_classes(), dtype=tf.int32)
        self.generator.trainable = True
        self.discriminator.trainable = True
        self.classifier.trainable = False
        display_input_dataset = self.get_dataset(Split.TEST, batch_size=DISPLAY_NUMBER, augment_data=False).take(1)
        display_input, _ = tf.data.experimental.get_single_element(display_input_dataset)
        test_dataset = self.get_dataset(Split.TEST, batch_size=batch_size, augment_data=False)
        train_dataset = self.get_dataset(Split.TRAIN, batch_size=batch_size, augment_data=False)
        def is_in_target_class(image, label):
            return tf.math.equal(label, target_class)
        target_dataset = self.get_dataset(Split.TRAIN, batch_size=batch_size, augment_data=False, filter=is_in_target_class)
        zipped_dataset = tf.data.Dataset.zip((train_dataset, target_dataset))
        self.plot_sample_output(display_input, 0)

        for epoch in range(epochs):
            start = time.time()
            for batches in zipped_dataset:
                train_batch, target_batch = batches
                train_image, _ = train_batch
                target_image, _ = target_batch
                self.train_step(train_image, target_image)
            self.accuracy_and_diff(test_dataset)
            self.write_stats(epoch)
            self.plot_sample_output(display_input, epoch + 1)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


    def write_stats(self, epoch):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('attack_accuracy', self.training_attack_acc.result(), step=epoch)
            tf.summary.scalar('diff_norm', self.training_diff_norm.result(), step=epoch)
            tf.summary.scalar('diff_norm_inf', self.training_diff_norm_inf.result(), step=epoch)
            tf.summary.scalar('gen_loss_mean', self.gen_loss_mean.result(), step=epoch)
            tf.summary.scalar('adv_loss_mean', self.adv_loss_mean.result(), step=epoch)
            tf.summary.scalar('hinge_loss_mean', self.hinge_loss_mean.result(), step=epoch)
            tf.summary.scalar('gen_total_loss_mean', self.gen_total_loss_mean.result(), step=epoch)
            tf.summary.scalar('discriminator_loss', self.discriminator_loss.result(), step=epoch)
        with self.test_summary_writer.as_default():
            tf.summary.scalar('attack_accuracy', self.test_attack_acc.result(), step=epoch)
            tf.summary.scalar('diff_norm', self.test_diff_norm.result(), step=epoch)
            tf.summary.scalar('diff_norm_inf', self.test_diff_norm_inf.result(), step=epoch)
            tf.summary.scalar('test_acc_clipped', self.test_acc_clipped.result(), step=epoch)
        template = 'epoch {}, train_acc: {}, test_acc: {}, train_diff: {}, test_diff: {}'
        print(template.format(epoch + 1,
                              self.training_attack_acc.result() * 100,
                              self.test_attack_acc.result() * 100,
                              self.training_diff_norm.result(),
                              self.test_diff_norm.result()))
        # self.training_attack_acc.reset_states()
        self.training_diff_norm.reset_states()
        self.gen_loss_mean.reset_states()
        self.adv_loss_mean.reset_states()
        self.hinge_loss_mean.reset_states()
        self.gen_total_loss_mean.reset_states()
        self.discriminator_loss.reset_states()
        self.test_attack_acc.reset_states()
        self.test_diff_norm.reset_states()

    def initialize_metrics(self):
        self.training_attack_acc = tf.keras.metrics.CategoricalAccuracy('training_attack_accuracy')
        self.test_attack_acc = tf.keras.metrics.CategoricalAccuracy('test_attack_accuracy')
        self.training_diff_norm = tf.keras.metrics.Mean('training_diff_norm', dtype=tf.float32)
        self.training_diff_norm_inf = tf.keras.metrics.Mean('training_diff_norm_inf', dtype=tf.float32)
        self.test_diff_norm = tf.keras.metrics.Mean('test_diff_norm', dtype=tf.float32)
        self.test_diff_norm_inf = tf.keras.metrics.Mean('test_diff_norm_inf', dtype=tf.float32)
        self.test_acc_clipped = tf.keras.metrics.CategoricalAccuracy('test_acc_clipped', dtype=tf.float32)
        self.gen_loss_mean = tf.keras.metrics.Mean('gen_loss_mean', dtype=tf.float32)
        self.adv_loss_mean = tf.keras.metrics.Mean('adv_loss_mean', dtype=tf.float32)
        self.hinge_loss_mean = tf.keras.metrics.Mean('hinge_loss_mean', dtype=tf.float32)
        self.gen_total_loss_mean = tf.keras.metrics.Mean('gen_total_loss_mean', dtype=tf.float32)
        self.discriminator_loss = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)
        train_log_dir = self.get_tensorboard_path() + '/train'
        test_log_dir = self.get_tensorboard_path() + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    @tf.function
    def train_step(self, train_images, target_images):
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            generated_noise = self.generator(train_images, training=True)
            adversary_images = generated_noise + train_images
            real_output = self.discriminator(target_images, training=True)
            fake_output = self.discriminator(adversary_images, training=True)
            classifier_output = self.classifier(adversary_images)
            gen_loss = self.alpha * self.generator.generator_loss(fake_output)
            adv_loss = tf.reduce_mean(self.classifier.loss(self.target_class, classifier_output))
            hinge_distance_loss =  self.beta * tf.math.reduce_mean(tf.math.maximum(0.0, tf.norm(generated_noise, axis=(1,2)) - self.c_constant))
            gen_total_loss = gen_loss + adv_loss + hinge_distance_loss
            # gen_total_loss = gen_loss
            disc_loss = self.alpha * self.discriminator.discriminator_loss(real_output, fake_output)
        diff_norm = tf.math.reduce_mean(tf.norm(generated_noise, axis=(1,2)), axis=0)
        diff_norm_inf = tf.math.reduce_mean(tf.norm(generated_noise, axis=(1,2), ord=np.inf), axis=0)
        gradients_of_generator = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


        # Update metrics
        self.training_attack_acc(self.target_class, classifier_output)
        self.gen_loss_mean(gen_loss)
        self.adv_loss_mean(adv_loss)
        self.hinge_loss_mean(hinge_distance_loss)
        self.gen_total_loss_mean(gen_total_loss)
        self.discriminator_loss(disc_loss)
        self.training_diff_norm(diff_norm)
        self.training_diff_norm_inf(diff_norm_inf)


    def accuracy_and_diff(self, test_dataset):
        for image, label in test_dataset:
            generated = self.generator(image)
            diff_norm = tf.math.reduce_mean(tf.norm(generated, axis=(1,2)), axis=0)
            diff_norm_inf = tf.math.reduce_mean(tf.norm(generated, axis=(1,2), ord=np.inf), axis=0)
            self.test_diff_norm(diff_norm)
            self.test_diff_norm_inf(diff_norm_inf)
            self.test_attack_acc(self.classifier(generated + image), self.target_class)
            self.test_acc_clipped(self.classifier(((generated+1.0)/2.0 + image)), self.target_class)



    def plot_sample_output(self, display_input, epoch):
        generated_images = self.generator(display_input)
        generated_images = generated_images + display_input
        probs = self.classifier(generated_images) if epoch >0 else self.classifier(display_input)
        figure = plt.figure(figsize=(10, 10))
        for i in range(generated_images.shape[0]):
            plt.subplot(3, 4, 2 * (i + 1) - 1)
            plt.imshow(generated_images[i, :, :, 0], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 4, 2 * (i + 1))
            plt.bar(np.arange(len(probs[i])), probs[i])
            plt.xticks(np.arange(len(probs[i])), np.arange(probs[i].numpy().size))
        # plt.show()
        with self.train_summary_writer.as_default():
            tf.summary.image("Training data", plot_to_image(figure), step=epoch)
        return figure