import time
import tensorflow as tf
from tensorflow_core.python.keras.metrics import CategoricalAccuracy
from tensorflow_datasets import Split
from models.Attacks.c_and_w import  carlini_wagner
from models.Attacks.deepfool import deepfool
from models.Attacks.fgsm import fgsm
from models.Attacks.gen_attack import gen_attack
from models.Attacks.map_elites import map_elites
from models.CIFAR10Models.ConvModel import ConvModel
from models.ImageNet.InceptionV3Wrapper import InceptionV3Wrapper
from models.ImageNet.ResNetWrapper import ResNetWrapper
from models.utils.images import show_plot

classifier = ResNetWrapper().load_model_data()

BATCH_SIZE = 1
target_class_int = 2
target_class = tf.one_hot(target_class_int, classifier.get_number_of_classes())

# UNTARGETED ATTACKS
metric = CategoricalAccuracy()
for data_sample in  classifier.get_dataset(Split.TEST, batch_size=1).take(5):
    image, labels = data_sample
    start = time.time()
    show_plot(classifier(tf.expand_dims(image[0], 0)), image[0], classifier.get_label_names())
    # ret_image, logits = deepfool(classifier, data_sample, max_iter=1000)
    ret_image, logits = fgsm(classifier, data_sample, target_class, eps=0.02)
    show_plot(logits[0], ret_image[0], classifier.get_label_names())
    metric.update_state(labels, logits)
    print("TIME: {} ACC: {}".format(str(time.time() - start), str(1.0-metric.result().numpy())))

# TARGETED ATTACKS
# accuracy = CategoricalAccuracy()
# for data_sample in classifier.get_dataset(Split.TEST, batch_size=BATCH_SIZE).take(10000):
#     image, labels = data_sample
#     start = time.time()
#     show_plot(classifier(tf.expand_dims(image[0], 0)), image[0], classifier.get_label_names())
    # ret_image, logits = carlini_wagner(classifier, data_sample, target_class, optimization_iter=1000, binary_iter=2, c_high=1.0)
    # ret_image, logits = fgsm(classifier, data_sample, target_class, eps=0.02)
    # show_plot(logits[0], ret_image[0], classifier.get_label_names())
    # accuracy.update_state(target_class, logits)
    # print("TIME: {:2f} ACC: {:2f}".format(time.time() - start, accuracy.result().numpy()))
#


# MAP-ELITES
# start = time.time()
# ret_image, logits = map_elites(model, iter_max=1000)
# for i in range(10):
#     show_plot(logits[i], ret_image[i], model.get_label_names())
# print("TIME: {:2f}".format(time.time() - start))

