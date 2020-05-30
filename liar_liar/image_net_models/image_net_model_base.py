import json
from os import path

from liar_liar.base_models.sequential_model import SequentialModel

import tensorflow as tf
import tensorflow_datasets as tfds

from liar_liar.image_net_models.image_net_dict import IMAGE_NET_DICT


class ImageNetModel(SequentialModel):

    def __init__(self, optimizer, loss, metrics, MODEL_NAME):
        super().__init__(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics,
                         MODEL_NAME=MODEL_NAME,
                         dataset_name='imagenet2012',
                         dataset_dir='D:\\')

    def get_input_shape(self):
        return (299,299,3)

    def get_dataset(self, split, batch_size=32, shuffle=10000, augment_data=True, **kwargs) -> tf.data.Dataset:
        def cast_labels(x, y):
            x = tf.cast(x, tf.float32)/255.0
            y = tf.one_hot(y, self.get_number_of_classes())
            x = tf.image.resize(x, (299, 299))
            x = tf.reshape(x, (299, 299, 3))
            return x, y
        dataset, info = tfds.load(self.dataset_name, split=tfds.Split.VALIDATION, with_info=True, as_supervised=True, data_dir=self.data_dir)  # type: tf.data.Dataset
        dataset = dataset.map(cast_labels).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        self.info = info
        self.test_steps = info.splits[tfds.Split.VALIDATION].num_examples // batch_size
        return dataset

    def get_label_names(self):
        labels = []
        for val in  IMAGE_NET_DICT.values():
            val = val[:20] if len(val) > 20 else val
            labels.append(val)
        return labels





def get_imagenet_models():
    """
    Returns:
        list[ImageNetModel]:
    """
    from liar_liar.image_net_models.inception_v3_wrapper import InceptionV3Wrapper
    from liar_liar.image_net_models.res_net_wrapper import ResNetWrapper
    return [
        ResNetWrapper(),
        InceptionV3Wrapper(),
    ]