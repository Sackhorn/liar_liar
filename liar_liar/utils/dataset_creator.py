import pickle

import tensorflow_datasets as tfds
import tensorflow as tf

def create_adv_dataset(dataset, attack_wrapper, model, params):
    """

    Returns:
        tf.data.Dataset:
    """
    dataset.repeat(0)
    new_images, new_labels = None, None
    attack = attack_wrapper(**params)
    for data_sample in dataset:
        images, labels = data_sample
        adv_image, _, _ = attack(model, data_sample)
        new_images = adv_image if new_images is None else tf.concat([new_images, adv_image], 0)
        new_labels = labels if new_labels is None else tf.concat([new_labels, labels], 0)
    adv_dataset = tf.data.Dataset.from_tensor_slices((new_images, new_labels))
    return adv_dataset

def save_adv_dataset(dataset, name):
    tf.data.experimental.save(dataset, name + ".tfrecord")
    print(dataset.element_spec)
    with open(name+".typespec", 'wb') as f:
        pickle.dump(dataset.element_spec, f)

def load_adv_dataset(name):
    with open(name+".typespec", 'rb') as f:
        type_spec = pickle.load(f)
    return tf.data.experimental.load(name + ".tfrecord", type_spec)
