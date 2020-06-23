import time
from abc import ABC

import numpy as np
from tensorflow.python.keras.metrics import CategoricalAccuracy

from liar_liar.utils.general_names import *
from liar_liar.utils.utils import batch_image_norm


class AttackMetric(ABC):

    def __init__(self):
        pass

    def accumulate(self, image, label, adv_image, logits, batch_size, target_class):
        pass


class AttackMetricsAccumulator():

    def __init__(self, metrics_list):
        self.metrics_list = metrics_list

    def accumulate_metrics(self, image, label, adv_image, logits, batch_size, target_class=None):
        current_batch_dict = {}
        for metric in self.metrics_list:
            current_batch = metric.accumulate(image,
                                              label,
                                              adv_image,
                                              logits,
                                              batch_size,
                                              target_class=target_class)
            current_batch_dict.update(current_batch)
        return current_batch_dict


class L2_Metrics(AttackMetric):

    def __init__(self):
        super().__init__()
        self.l2_distance = np.array([])

    def accumulate(self, image, label, adv_image, logits, batch_size, target_class):
        self.l2_distance = np.append(self.l2_distance, batch_image_norm(image - adv_image).numpy().flatten())
        return {
            L2_AVERAGE_KEY : np.mean(self.l2_distance),
            L2_MEDIAN_KEY : np.median(self.l2_distance)
        }


class Accuracy(AttackMetric):

    def __init__(self):
        super().__init__()
        self.accuracy = CategoricalAccuracy()

    def accumulate(self, image, label, adv_image, logits, batch_size, target_class):
        if target_class is None:
            self.accuracy.update_state(label, logits)
            return {ACCURACY_KEY: 1.0 - self.accuracy.result().numpy()} #TODO: This is wrong for deepfool with imagenet
        else:
            self.accuracy.update_state(target_class, logits)
            return {ACCURACY_KEY : self.accuracy.result().numpy()}


class Robustness(AttackMetric):
    def __init__(self):
        super().__init__()
        self.robustness = np.array([])

    def accumulate(self, image, label, adv_image, logits, batch_size, target_class):
        image_norm = batch_image_norm(image).numpy().flatten()
        adv_image_norm = (batch_image_norm(adv_image) - batch_image_norm(image)).numpy().flatten()
        batch_robustness = np.divide(adv_image_norm, image_norm)
        self.robustness = np.append(self.robustness, batch_robustness)
        return {ROBUSTNESS_KEY : np.mean(self.robustness)}

class Timing(AttackMetric):
    def __init__(self):
        super().__init__()
        self.time = time.time()
        self.time_per_sample = np.array([])

    def accumulate(self, image, label, adv_image, logits, batch_size, target_class):
        time_per_batch = time.time() - self.time
        self.time_per_sample = np.append(self.time_per_sample, [time_per_batch / batch_size])
        self.time = time.time()

        return {
            TIME_PER_BATCH_KEY : time_per_batch,
            AVG_TIME_SAMPLE_KEY : np.mean(self.time_per_sample)
        }