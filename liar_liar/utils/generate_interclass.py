import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow_datasets import Split

from liar_liar.attacks.bfgs import bfgs_wrapper
from liar_liar.attacks.c_and_w import carlini_wagner_wrapper
from liar_liar.attacks.fgsm import fgsm_targeted_wrapper
from liar_liar.attacks.gen_attack import gen_attack_wrapper
from liar_liar.base_models.model_names import *
from liar_liar.base_models.sequential_model import get_all_models, SequentialModel
from liar_liar.utils.general_names import *
from liar_liar.utils.general_names import PARAMETERS_KEY
from liar_liar.utils.utils import disable_tensorflow_logging, find_or_create_file_path


def generate_targeted_attack_grid(attack_params, attack_wrapper, file_name, retries=20):
    disable_tensorflow_logging()
    all_models = get_all_models()
    classifier: SequentialModel
    for classifier in all_models:
        try:
            model_dict = attack_params[classifier.MODEL_NAME]
        except KeyError:
            print("classifier model not found in interclass params dict")
            continue
        parameters = model_dict[PARAMETERS_KEY]

        for parameter_set in parameters:
            attack = attack_wrapper(**parameter_set)
            nmb_classes = classifier.get_number_of_classes()
            nmb_classes = 10 if nmb_classes >= 10 else nmb_classes
            true_class_dict = {}
            for true_class in range(nmb_classes):
                true_class_one_hot = tf.one_hot(true_class, classifier.get_number_of_classes())
                def get_baseclass_not_misclassfied(image, label):
                    image = tf.expand_dims(image, 0)
                    classification = tf.one_hot(tf.argmax(classifier(image), 1), classifier.get_number_of_classes())
                    classified_fine = tf.math.reduce_all(tf.math.equal(classification, label))
                    in_true_class = tf.math.reduce_all(tf.math.equal(true_class_one_hot, label))
                    ret_val = tf.math.logical_and(classified_fine, in_true_class)
                    return ret_val
                dataset = classifier.get_dataset(Split.TEST,
                                                 batch_size=1,
                                                 shuffle=10,
                                                 filter=get_baseclass_not_misclassfied)
                range_nmb_target_classes = list(range(nmb_classes))
                range_nmb_target_classes.remove(true_class)
                true_class_dict[true_class] = {}
                for data_sample in dataset.take(retries):
                    for target_class in range_nmb_target_classes:
                        target_class_one_hot = tf.one_hot(target_class, classifier.get_number_of_classes())
                        found_sample = False
                        true_class_dict[true_class][true_class] = data_sample[0]
                        ret_image, logits, _ = attack(classifier, data_sample, target_class_one_hot)
                        if tf.math.reduce_all(tf.math.equal(tf.argmax(logits, 1), tf.argmax(target_class_one_hot))):
                            found_sample = True
                            true_class_dict[true_class][target_class] = ret_image
                        if not found_sample:
                            break
                    if found_sample:
                        break
            if not found_sample:
                print("run out of retries for {}".format(classifier.MODEL_NAME))
                continue

            show_plot_target_class_grid(true_class_dict,
                                        file_name + classifier.MODEL_NAME + ".png",
                                        classifier.get_label_names())

def show_plot_target_class_grid(images_dict, file_name, label_names):
    """
    Args:
        images_dict (dict):
    """
    file_name = find_or_create_file_path(file_name)
    figure: Figure = plt.figure(figsize=(20, 20))
    grid = ImageGrid(figure, 111, nrows_ncols=(10, 10), axes_pad=0.1)
    image_arr = []
    for true_class in  range(10):
        for target_class in range(10):
            image_arr.append(tf.squeeze(images_dict[true_class][target_class]))
    for i, (ax, im) in enumerate(zip(grid, image_arr)):
        ax.imshow(im, cmap=plt.get_cmap("gray"))
        y_label = label_names[i // 10]
        x_label = label_names[i % 10]
        ax.set_ylabel(y_label, fontsize=20)
        ax.set_xlabel(x_label, fontsize=20)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    figure.tight_layout()
    figure.savefig(file_name)
    # plt.savefig(file_name)

def generate_grid_bfgs():
    bfgs_params = [{ITER_MAX: 10000}]
    bfgs_model_params = {
        MNIST_CONV_NAME: {PARAMETERS_KEY: bfgs_params},
        CIFAR_10_CONV_NAME: {PARAMETERS_KEY: bfgs_params},
        CIFAR_100_CONV_NAME: {PARAMETERS_KEY: bfgs_params},
        INCEPTION_V3_NAME: {PARAMETERS_KEY: bfgs_params}}
    generate_targeted_attack_grid(bfgs_model_params, bfgs_wrapper, "../../latex/img/grid_bfgs_", retries=5)

def generate_grid_llfgsm():
    llfgsm_model_params = {
        MNIST_CONV_NAME:{PARAMETERS_KEY: [{ITER_MAX: 1000, EPS: 0.0005}],},
        CIFAR_10_CONV_NAME:{PARAMETERS_KEY: [{ITER_MAX: 100, EPS: 0.0005}],},
        CIFAR_100_CONV_NAME:{PARAMETERS_KEY: [{ITER_MAX: 100, EPS: 0.0001}],},
        INCEPTION_V3_NAME:{PARAMETERS_KEY: [{ITER_MAX: 100, EPS: 0.0001}],}
    }
    generate_targeted_attack_grid(llfgsm_model_params, fgsm_targeted_wrapper, "../../latex/img/grid_llfgsm_", retries=10)

def generate_grid_genattack():
    gen_attack_params = [{GENERATION_NUMBER:10000, POPULATION_NMB:6, DELTA:0.05, MUTATION_PROBABILITY:0.05}]
    genattack_params_models = {
        CIFAR_10_CONV_NAME :{PARAMETERS_KEY : gen_attack_params,},
        CIFAR_100_CONV_NAME:{PARAMETERS_KEY : gen_attack_params,},
        MNIST_CONV_NAME:{PARAMETERS_KEY : gen_attack_params,},
        INCEPTION_V3_NAME:{PARAMETERS_KEY : gen_attack_params,}}

    generate_targeted_attack_grid(genattack_params_models, gen_attack_wrapper, "../../../latex/img/grid_genattack_", retries=5)

def generate_grid_carlini_wagner():
    carliniwagner_params = [{OPTIMIZATION_ITER:1000, BINARY_ITER:10, C_HIGH:100.0, C_LOW:0.0, KAPPA:0.0},]
    carliniwagner_params_models = {
        CIFAR_10_CONV_NAME :{PARAMETERS_KEY : carliniwagner_params,},
        CIFAR_100_CONV_NAME:{PARAMETERS_KEY : carliniwagner_params,},
        MNIST_CONV_NAME:{PARAMETERS_KEY : carliniwagner_params,},
        INCEPTION_V3_NAME:{PARAMETERS_KEY : carliniwagner_params,}}

    generate_targeted_attack_grid(carliniwagner_params_models, carlini_wagner_wrapper, "../../../latex/img/grid_carlini_", retries=5)

if __name__ == "__main__":
    # generate_grid_llfgsm()
    # generate_grid_bfgs()
    generate_grid_carlini_wagner()
