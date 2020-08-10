import abc
from liar_liar.models.base_models.sequential_model import SequentialModel
import tensorflow as tf


class Attack(abc.ABC):

    @abc.abstractmethod
    def init_wrapper(self, *args, **kwargs):
        self._wrapper = None

    def __init__(self, *args, **kwargs):
        self.init_wrapper(*args, **kwargs)
        self.__name__ = self._wrapper.__name__

    def __call__(self, model, data_sample, target_class=None):
        """
        As for untargeted attacks pass None as target_class
        :rtype: (tf.Tensor, tf.Tensor, dict)
        :type data_sample: (tf.Tensor, tf.Tensor)
        :type model: SequentialModel

        """
        if target_class is None:
            return self._wrapper(model, data_sample)
        else :
            return self._wrapper(model, data_sample, target_class)




