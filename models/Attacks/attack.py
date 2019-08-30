import abc
import models.BaseModels.SequentialModel
import tensorflow as tf


class Attack(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def run_attack(model, dataset, target_class):
        """
        As for untargeted attacks pass None as target_class
        :rtype: (EagerTensor, int)
        :type dataset: tf.data.Dataset
        :type model: SequentialModel

        """
        pass



