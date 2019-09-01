import abc
import models.BaseModels.SequentialModel
import tensorflow as tf


class Attack(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def run_attack(model, data_sample, target_class):
        """
        As for untargeted attacks pass None as target_class
        :rtype: (EagerTensor, int)
        :type data_sample: tf.data.Dataset
        :type model: SequentialModel

        """
        pass



