import time
import tensorflow as tf
from tensorflow_core.python.keras.metrics import CategoricalAccuracy
from tensorflow_datasets import Split

from liar_liar.attacks import map_elites
from liar_liar.attacks.bfgs import bfgs_wrapper
from liar_liar.attacks.c_and_w import carlini_wagner_wrapper
from liar_liar.attacks.deepfool import deepfool_wrapper
from liar_liar.attacks.fgsm import fgsm_targeted_wrapper, untargeted_fgsm_wrapper
from liar_liar.attacks.gen_attack import gen_attack_wrapper
from liar_liar.attacks.jsma import jsma_targeted_wrapper
from liar_liar.cifar_10_models.cifar_10_conv_model import CIFAR10ConvModel
from liar_liar.utils.images import show_plot


def run_test(classifier, attack, targeted, batch_size, nmb_elements, target_class=None, show_plots=True):
    """

    Args:
        classifier: A classifier we attack
        attack: A wrapped attack method
        targeted: wether the attack is targeted or not
        batch_size: number of elements being put at once into the attack
        nmb_elements: number of batches we want to run through attack
    """
    accuracy = CategoricalAccuracy()
    for data_sample in classifier.get_dataset(Split.TEST, batch_size=batch_size).take(nmb_elements): #TODO: Prune misclassified elements from dataset and elements that already are in target class
        image, labels = data_sample
        start = time.time()
        if targeted:
            ret_image, logits = attack(classifier, data_sample, target_class)
            accuracy.update_state(target_class, logits)
            accuracy_result = accuracy.result().numpy()
        else:
            ret_image, logits = attack(classifier, data_sample)
            accuracy.update_state(labels, logits)
            accuracy_result = 1.0 - accuracy.result().numpy()
        if show_plots:
            show_plot(classifier(tf.expand_dims(image[0], 0)), image[0], classifier.get_label_names(), plot_title=attack.__name__)
            show_plot(logits[0], ret_image[0], classifier.get_label_names(), plot_title=attack.__name__)
        print("TIME: {:2f} ATACK_ACC: {:2f}".format(time.time() - start, accuracy_result))
    return accuracy_result

def run_classifier_tests(classifier, attack_list, targeted, batch_size, nmb_elements, target_class=None, show_plots=True):
    acc_arr = []
    for attack in attack_list:
        acc_result = run_test(classifier, attack, targeted, batch_size, nmb_elements, target_class=target_class, show_plots=show_plots)
        acc_arr.append(acc_result)
        print("Attack {} Attack Accuracy {:2f}".format(attack.__name__, acc_result))
    for attack, accuracy in zip(attack_list, acc_arr):
        print("Attack {} Attack Accuracy {:2f}".format(attack.__name__, accuracy))

def map_elites_test(classifier, iter_max):
    start = time.time()
    ret_image, logits = map_elites.map_elites(classifier, iter_max=iter_max)
    for i in range(10):
        show_plot(logits[i], ret_image[i], classifier.get_label_names(), plot_title='map_elites')
    print("TIME: {:2f}".format(time.time() - start))



cifar10_attacks_targeted = []
# cifar10_attacks_targeted.append(fgsm_targeted_wrapper(iter_max=100, eps=0.001))
# cifar10_attacks_targeted.append(bfgs_wrapper(iter_max=1000))
# cifar10_attacks_targeted.append(carlini_wagner_wrapper(optimization_iter=100, binary_iter=10))
cifar10_attacks_targeted.append(gen_attack_wrapper(generation_nmb=100000, delta=0.05))
# cifar10_attacks_targeted.append(jsma_targeted_wrapper())

cifar10_attacks_untargeted = []
cifar10_attacks_untargeted.append(untargeted_fgsm_wrapper(eps=0.1))
cifar10_attacks_untargeted.append(deepfool_wrapper(max_iter=10000))

classifier = CIFAR10ConvModel().load_model_data()
BATCH_SIZE = 1
NMB_ELEMENTS = 1000
target_class_int = 5
target_class = tf.one_hot(target_class_int, classifier.get_number_of_classes())
# map_elites_test(classifier, 1000)
run_classifier_tests(classifier,
                     cifar10_attacks_targeted,
                     targeted=True,
                     batch_size=BATCH_SIZE,
                     nmb_elements=NMB_ELEMENTS,
                     target_class=target_class,
                     show_plots=False)
# run_classifier_tests(classifier, cifar10_attacks_untargeted, targeted=False, batch_size=BATCH_SIZE, nmb_elements=NMB_ELEMENTS)
