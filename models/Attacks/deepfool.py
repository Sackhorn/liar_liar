from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python import enable_eager_execution
from tensorflow.python.eager.context import DEVICE_PLACEMENT_SILENT

from models.CIFAR10Models.ConvModel import ConvModel
from models.ImageNet.InceptionV3Wrapper import ResNetWrapper
from models.utils.images import show_plot
from models.MNISTModels.DenseModel import DenseModel
from models.BaseModels.SequentialModel import SequentialModel
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# TODO: Generalize for all models
# Source https://arxiv.org/pdf/1511.07528.pdf


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
        for k in range(nmb_classes):
            with tf.GradientTape(persistent=False) as tape:
                tape.watch(image)
                logits = model(image, get_raw=False)
                gradients_by_cls.append(tf.squeeze(tape.gradient(logits[0][k], image)).numpy())

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


def test_deepfool():
    model = ResNetWrapper()
    # model = ConvModel()
    # model.load_model_data()
    eval_dataset = model.get_dataset(tfds.Split.TEST, batch_size=1)
    target_label = 5
    for data_sample in eval_dataset.take(5):
        deepfool(data_sample, model)

if __name__ == "__main__":
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    # enable_eager_execution(config=config)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    enable_eager_execution()
    test_deepfool()
