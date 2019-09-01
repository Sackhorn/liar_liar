import time

from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python import enable_eager_execution

from models.Attacks.attack import Attack
from models.ImageNet import InceptionV3Wrapper
from models.ImageNet.InceptionV3Wrapper import ResNetWrapper
from models.utils.images import show_plot
from models.BaseModels.SequentialModel import SequentialModel
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# TODO: Generalize for all models
# Source https://arxiv.org/pdf/1511.07528.pdf

class DeepFool(Attack):


    @staticmethod
    def deepfool(data_sample, model, max_iter=10, min=0.0, max=1.0):
        """

        :type model: SequentialModel
        """

        nmb_classes = model.get_number_of_classes()
        image, label = data_sample
        label = tf.argmax(tf.squeeze(label)).numpy()
        show_plot(model(image), image, model.get_label_names())
        iter = 0
        while tf.argmax(tf.squeeze(model(image))).numpy() == label and iter < max_iter:
            logits = None
            gradients_by_cls = []
            logits_list = []
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(image)
                logits = model(image)
                for k in range(nmb_classes):
                    logits_list.append(logits[0][k])
            for k in range(nmb_classes):
                gradient = tape.gradient(logits_list[k], image)
                gradients_by_cls.append(tf.squeeze(gradient).numpy())
            del tape
            logits = tf.squeeze(logits).numpy()
            w_prime = []
            f_prime = []
            for k in range(nmb_classes):
                if k == tf.squeeze(label).numpy():
                    f_prime.append(float('+inf'))
                    w_prime.append(float("+inf"))
                    continue
                w_prime.append(gradients_by_cls[k] - gradients_by_cls[label])
                f_prime.append(logits[k] - logits[label])
            tmp = []
            for k in range(nmb_classes):
                if k == tf.squeeze(label).numpy():
                    tmp.append(float("+inf"))
                    continue
                tmp.append(abs(f_prime[k]) / np.linalg.norm(w_prime[k]))
            l = np.argmin(tmp)
            perturbation = (abs(f_prime[l]) * w_prime[l]) / np.square(np.linalg.norm(w_prime[l]))
            new_image = image.numpy() + perturbation
            image = tf.convert_to_tensor(new_image, dtype=tf.float32)
            image = tf.clip_by_value(image, min, max)
            iter += 1
            print("iteration: " + str(iter))
        show_plot(model(image), image, model.get_label_names())

    @staticmethod
    def run_attack(model, data_sample):
        for data_sample in data_sample:
            DeepFool.deepfool(data_sample, model)

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("setting memory growth at: " + str(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    enable_eager_execution()

    model = ResNetWrapper()
    dataset = model.get_dataset(tfds.Split.VALIDATION, batch_size=1, shuffle=2).take(2)

    DeepFool.run_attack(model, dataset)
