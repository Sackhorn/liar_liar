import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.python.data.ops.dataset_ops import DatasetV1

from models.BaseModels.SequentialModel import SequentialModel


class CIFAR10Model(SequentialModel):

    info: tfds.core.DatasetInfo

    def __init__(self, optimizer, loss, metrics, MODEL_NAME=""):
        super().__init__(nmb_classes=10,
                         optimizer=optimizer,
                         loss=loss,
                         metrics=metrics,
                         MODEL_NAME=MODEL_NAME)

    def get_dataset(self, split, name='cifar10', batch_size=32, shuffle=10000):
        dataset =  super().get_dataset(split, name, batch_size, shuffle)
        dataset = dataset.map(lambda x,y: (tf.div(x, 255), y))
        return dataset