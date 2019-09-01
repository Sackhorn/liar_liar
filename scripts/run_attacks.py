import time
import tensorflow_datasets as tfds

from multiprocessing.pool import Pool
from math import floor
from models.Attacks.jsma import jsma_plus_increasing
from models.CIFAR10Models.ConvModel import ConvModel


def run_targeted_atacks(attack, model, dataset_percentage):
    succesful_attacks = 0
    dataset = model.get_dataset(tfds.Split.TEST, batch_size=1, shuffle=1)
    test_samples_nmb = floor(model.get_info().splits['test'].num_examples * dataset_percentage)
    for data_sample in dataset.take(test_samples_nmb):
        start = time.time()
        image, true_class = data_sample
        true_class = true_class.numpy().squeeze().argmax()
        target_class = true_class - 1 % model.get_number_of_classes()
        if true_class != model(image).numpy().squeeze().argmax():
            continue
        image, return_class = attack.run_attack(model, data_sample, target_class)
        succesful_attacks += 1 if return_class == target_class else 0
        print(succesful_attacks)
        print("it took: " + str(time.time() - start))
    failed_attacks = test_samples_nmb - succesful_attacks
    return succesful_attacks, failed_attacks

def run_atacks_multiproccess(attack, model, dataset_percentage, target_class=None):
    dataset = model.get_dataset(tfds.Split.TEST, batch_size=1, shuffle=1)
    test_samples_nmb = floor(model.get_info().splits['test'].num_examples * dataset_percentage)
    attack = wrap_attack_for_multiprocess(attack, model, target_class)
    with Pool(32) as p:
        ret_array = p.map(attack, dataset.take(test_samples_nmb))

    print("successfully attacked " + str(ret_array.count(True)) + " out of " + str(test_samples_nmb - ret_array.count(None)))
    print("win ration is " + str((ret_array.count(True)/(test_samples_nmb - ret_array.count(None))) * 100.0) )


def wrap_attack_for_multiprocess(attack, model, target_class=None):
    def wrapper(data_sample):
        start = time.time()
        image, true_class = data_sample
        classified_class = model(image).numpy().squeeze().argmax()
        true_class = true_class.numpy().squeeze().argmax()
        # this covers a case when dataset gives us a misclassified example or one that agrees with target_class
        if true_class != classified_class or true_class == target_class:
            return (None, -1.0, -1) # (was_attack_successful, time_it_took, nmb_iterations)
        _, return_class, iterations = attack().run_attack(model, data_sample, target_class)
        duration = time.time() - start
        was_successful = return_class == target_class
        return (was_successful, duration, iterations)

    return wrapper

if __name__ == "__main__":

    start = time.time()
    run_atacks_multiproccess(jsma_plus_increasing, ConvModel().load_model_data(), 1.0, target_class=0)
