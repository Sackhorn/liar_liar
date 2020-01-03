import time
import tensorflow as tf
import numpy as np
from tensorflow_datasets import Split

from models.Attacks.GenAttack import GenAttack
from models.Attacks.GenAttackVectorized import GenAttackVectorized
from models.Attacks.c_and_w import CarliniWagner
from models.Attacks.deepfool import DeepFool
from models.Attacks.fgsm import FGSM
from models.Attacks.jsma import jsma_plus_increasing
from models.CIFAR10Models.ConvModel import ConvModel
from models.ImageNet.InceptionV3Wrapper import InceptionV3Wrapper
from models.utils.images import show_plot
from models.utils.utils import count

model = ConvModel().load_model_data()
# model = ResNetWrapper().load_model_data()
# attacks = [FGSM, DeepFool, CarliniWagner, jsma_plus_increasing]
attacks = [GenAttackVectorized]
# attacks = [GenAttack]

target_class = tf.one_hot(2, model.get_number_of_classes())
avg = []
for data_sample in  model.get_dataset(Split.TEST, batch_size=20).take(100):
    image, labels = data_sample

    #this checks if we don't have a misclassified example in batch
    if tf.reduce_all(
        tf.reduce_all(
            tf.math.not_equal(
                tf.one_hot(tf.argmax(model(image), axis=1), model.get_number_of_classes()),
                labels), axis=1), axis=0):
        continue


    for attack in attacks:
        start = time.time()
        print("ATTACK: " + attack.__name__)
        ret_image, logits = attack.run_attack(model, data_sample, target_class)
        avg.append(count(model, ret_image, target_class))
        print("COUNT: " + str(count(model, ret_image, target_class)))
        print("TIME: " + str(time.time() - start))
    print(np.average(np.array(avg)))
