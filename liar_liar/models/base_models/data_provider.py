import os
import sys

import tensorflow as tf
import tensorflow_datasets as tfds

from datetime import datetime
from os.path import *

from liar_liar.utils.utils import get_gcs_path


class DataProvider():

    ROOT_DIR = get_gcs_path() if get_gcs_path() is not None else join(dirname(__file__), os.pardir, os.pardir, os.pardir)
    MODEL_DIR_NAME = "saved_models"
    TENSORBOARD_NAME = "tboardlog"

    def __init__(self):
        pass

    def register_data_provider(self, MODEL_NAME, dataset_name, dataset_dir):
        if MODEL_NAME == "" or dataset_name=="":
            raise NotImplementedError("Please support model and dataset name in subclass super call to this base class")
        self.MODEL_NAME = MODEL_NAME
        self.SAVE_DIR = join(DataProvider.ROOT_DIR, DataProvider.MODEL_DIR_NAME, self.MODEL_NAME)
        self.dataset_name = dataset_name
        self.data_dir = dataset_dir
        self.builder = tfds.builder(self.dataset_name)

    def get_tensorboard_path(self):
        date_string = datetime.today().strftime("%d_%m_%Y_%H_%M")
        return join(DataProvider.ROOT_DIR, DataProvider.TENSORBOARD_NAME, self.MODEL_NAME, date_string)

    def call(self, input):
        raise NotImplementedError("Implement call when overriding ModelBase")

    def get_input_shape(self):
        return self.builder.info.features['image'].shape

    def get_number_of_classes(self):
        return self.builder.info.features['label'].num_classes

    def get_label_names(self):
        return self.builder.info.features['label'].names

    def get_dataset(self, split, batch_size=32, shuffle=10000, augment_data=True, filter=None) -> tf.data.Dataset:
        augmentations = [rotate, flip]

        def cast_labels(x, y):
            x = tf.cast(x, tf.float32)/255.0
            y = tf.one_hot(y, self.get_number_of_classes())
            return x, y

        def augment_data(x, y):
            if x.shape[3] > 1 and color not in augmentations : augmentations.append(color)
            for f in augmentations:
                x = tf.cond(tf.random.uniform([], minval=0.0, maxval=1.0) > 0.75, lambda: f(x), lambda: x)
            tf.clip_by_value(x, 0, 1)
            return x, y
        # if self.dataset_name in ['mnist', 'cifar10', 'cifar100'] and "gs:" not in self.data_dir:
        if self.dataset_name in ['mnist', 'cifar10', 'cifar100']:
            self.data_dir = None
        if self.dataset_name == "imagenet2012":
            split = tfds.Split.VALIDATION
        dataset, info = tfds.load(self.dataset_name, split=split, with_info=True, as_supervised=True, data_dir=self.data_dir)  # type: tf.data.Dataset
        dataset = dataset.map(cast_labels)
        if filter is not None:
            dataset = dataset.filter(filter)
        dataset = dataset.shuffle(shuffle).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
        if split == tfds.Split.TRAIN and augment_data:
            dataset = dataset.map(augment_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.info = info
        self.batch_size = batch_size
        self.train_steps_per_epoch = info.splits['train'].num_examples // batch_size
        try:
            self.test_steps = info.splits['test'].num_examples // batch_size
        except:
            self.test_steps = info.splits[tfds.Split.VALIDATION].num_examples // batch_size
        return dataset

    def get_info(self):
        return self.info

    def load_model_data(self):
        try:
            self.load_weights(self.SAVE_DIR)
        except:
            print("Failed to load model from file: " + self.SAVE_DIR)
        return self

    def save_model_data(self):
        self.save_weights(self.SAVE_DIR)
        print("Successfully saved model data to: " + self.SAVE_DIR)

def flip(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x

def color(x):
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def rotate(x):
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))