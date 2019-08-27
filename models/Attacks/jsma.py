import itertools
import time

from tensorflow.python import enable_eager_execution, Tensor
from models.CIFAR10Models.ConvModel import ConvModel
from models.ImageNet.InceptionV3Wrapper import ResNetWrapper
from models.MNISTModels.DenseModel import DenseModel
from models.utils.images import show_plot
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# TODO: Generalize for all models
# Source https://arxiv.org/pdf/1511.07528.pdf


def jsma(data_sample,
         model,
         i_max,
         theta,
         target_label=-1,
         is_increasing=True,
         min=0.0,
         max=1.0,
         show_plots=False,
         use_logits=False):

    is_targeted = target_label != -1
    image, true_label = data_sample
    true_label = true_label.numpy().squeeze().argmax()
    current_prediction = model(image).numpy().squeeze().argmax()
    if true_label != current_prediction:
        raise ValueError("The image given was wrongly classified in the first place")

    input_shape = model.get_input_shape()
    all_pixels = generate_all_pixels(input_shape)
    # all_pixels = prune_saturated_pixels(all_pixels, image, is_increasing, min, max)
    iter = 0

    if show_plots:
        show_plot(model(image), image, labels_names=model.get_label_names())

    # this function determines whether the algorithm should end work
    def has_met_target(is_targeted, target_label, true_label, current_prediction):
        if is_targeted:
            return current_prediction == target_label
        else:
            return current_prediction != true_label

    while iter < i_max and len(all_pixels) > 0 and not has_met_target(is_targeted, target_label, true_label, current_prediction):
        start = time.time()
        chosen_pixel_pair, theta_sign = saliency_map(model,
                                                     image,
                                                     true_label,
                                                     target_label,
                                                     all_pixels,
                                                     is_targeted,
                                                     is_increasing,
                                                     use_logits)
        add_tensor = np.zeros(input_shape)
        first, second = chosen_pixel_pair
        add_tensor[first] = theta * theta_sign
        add_tensor[second] = theta * theta_sign

        add_tensor = add_tensor.reshape(input_shape)
        add_tensor = tf.convert_to_tensor(add_tensor, dtype=tf.float32)
        image = tf.clip_by_value(image + add_tensor, min, max)

        current_prediction = model(image).numpy().squeeze().argmax()
        all_pixels = prune_saturated_pixels(all_pixels, image, is_increasing, min, max)
        iter += 1
        if show_plots:
            show_plot(model(image), image, labels_names=model.get_label_names())
            # print("iteration: " + str(iter) + " took " + str(time.time() - start))


    return image

# TODO:nie traktować każdej barwy jako osobny ficzer tylko sumować i zmieniać intensywność
def saliency_map(model, image, true_label, target_label, all_pixels, is_targeted, is_increasing, use_logits):


    nmb_classes = model.get_number_of_classes()


    # Generate forward derivatives for each of the classes
    grds_by_cls = []
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(image)
        logits = model(image, get_raw=use_logits)
        for k in range(nmb_classes):
            grds_by_cls.append(tape.gradient(logits[0][k], image).numpy().squeeze())

    grds_by_cls = np.array(grds_by_cls)
    first_pix, second_pix = generate_test(all_pixels)

    # This is important in case of un-targeted attack when we want to consider all classes except the true one
    # not just the target_class
    classes_to_iterate_over = [target_label] if is_targeted else list(range(nmb_classes))
    max_value = 0.0
    best_pair = None

    # for target_class in classes_to_iterate_over:
    alpha = grds_by_cls[target_label, first_pix[:, 0], first_pix[:, 1], first_pix[:, 2]] + grds_by_cls[target_label, second_pix[:, 0], second_pix[:, 1], second_pix[:, 2]]
    beta = grds_by_cls[:, first_pix[:, 0], first_pix[:, 1], first_pix[:, 2]] + grds_by_cls[:, second_pix[:, 0], second_pix[:, 1], second_pix[:, 2]]
    beta[target_label] = 0.0
    beta = np.sum(beta, axis=0)

    # This way we exclude pixels that don't met our criteria
    if (is_increasing and is_targeted) or (not is_increasing and not is_targeted):
        alpha[alpha < 0] = np.nan
        beta[beta > 0] = np.nan
    elif not (is_increasing and is_targeted) or (is_increasing and not is_targeted):
        alpha[alpha > 0] = np.nan
        beta[beta < 0] = np.nan

    pixes = np.multiply(alpha, beta) * -1
        # if pixes.max() > max_value:
        #     best_pair = np.nanargmax(pixes)
        #     max_value = pixes.max()


    best_pair = np.nanargmax(pixes)
    return_pixel_pair = (tuple(first_pix[best_pair].squeeze()), tuple(second_pix[best_pair].squeeze()))

    # input_shape = model.get_input_shape()
    # all_pixel_pairs = generate_all_pixel_pairs(all_pixels)
    # initialize before algorithm run
    # max = float("-inf")
    # return_pixel_pair = None
    #
    # for considered_class in classes_to_iterate_over:
    #     for pixel_pair in all_pixel_pairs:
    #         first, second = pixel_pair
    #         beta = 0.0
    #
    #         alpha = grds_by_cls[considered_class][first] + grds_by_cls[considered_class][second]
    #         for class_nmb in range(nmb_classes):
    #             if class_nmb != considered_class:
    #                 beta += grds_by_cls[class_nmb][first] + grds_by_cls[class_nmb][second]
    #
    #         if (is_increasing and is_targeted) or (not is_increasing and not is_targeted):
    #             is_pair_accepted = alpha > 0.0 and beta < 0 and -alpha * beta > max
    #         elif not (is_increasing and is_targeted) or (is_increasing and not is_targeted):
    #             is_pair_accepted = alpha < 0.0 and beta > 0 and -alpha * beta > max
    #
    #         if is_pair_accepted:
    #             return_pixel_pair = pixel_pair
    #             max = -alpha*beta

    theta_sign = 1 if is_increasing else -1
    return return_pixel_pair, theta_sign



def generate_all_pixels(shape):
    start = time.time()
    x, y, z = shape
    all_pixels = []
    for i in range(x):
        for j in range(y):
            if z > 1:
                for k in range(z):
                    all_pixels.append([i, j, k])
            else:
                all_pixels.append([i, j])
    # print("generation of pixels took " + str(time.time() - start))
    return all_pixels

def prune_saturated_pixels(all_pixels, image, is_increasing, min, max):
    np_image = image.numpy().squeeze()
    pixels_to_remove = []
    for pixel in all_pixels:
        if is_increasing and (max-np_image[tuple(pixel)]) < 1e-5:
            pixels_to_remove.append(pixel)
        elif not is_increasing and (np_image[tuple(pixel)]-min) < 1e-5:
            pixels_to_remove.append(pixel)
    for pixel in pixels_to_remove:
        all_pixels.remove(pixel)
    return all_pixels

def pixels_to_remove(all_pixels, image, is_increasing, min, max):
    np_image = image.numpy().squeeze()
    pixels_to_remove = []

    for pixel in all_pixels:
        if is_increasing and (max-np_image[tuple(pixel)]) < 1e-5:
            pixels_to_remove.append(pixel)
        elif not is_increasing and (np_image[tuple(pixel)]-min) < 1e-5:
            pixels_to_remove.append(pixel)

    return pixels_to_remove

def generate_all_pixel_pairs(all_pixels):
    all_pairs = []
    for i in range(len(all_pixels)):
        first = all_pixels[i]
        for j in all_pixels[i+1:]:
            second = j
            all_pairs.append((first, second))
    return all_pairs

def generate_test(all_pixels):
    start = time.time()

    all_pixels = np.array(all_pixels)
    shape = all_pixels.shape
    indices = np.column_stack(np.triu_indices(np.prod(shape[0]), 1))
    first = all_pixels[indices[:,0]]
    second = all_pixels[indices[:,1]]
    # print("generating pairs took: " + str(time.time() - start))
    # shape = 32,32,3
    # pairs = np.column_stack(np.unravel_index(np.arange(np.prod(shape)),shape))[np.column_stack(np.triu_indices(np.prod(shape),1))]
    # first = pairs[:,0,:]
    # second = pairs[:,1,:]

    # first_pix = []
    # second_pix = []
    # for i in range(len(all_pixels)):
    #     first = all_pixels[i]
    #     for j in all_pixels[i+1:]:
    #         second = j
    #         first_pix.append(first)
    #         second_pix.append(second)
    # return np.array(first_pix), np.array(second_pix)
    return first, second

def targeted_jsma(data_sample, model, i_max, eps, target_label, is_increasing=True, min=0.0, max=1.0, use_logits=False, show_plots=False):
    return jsma(data_sample, model, i_max, eps, target_label=target_label, is_increasing=is_increasing, use_logits=use_logits, min=min, max=max, show_plots=show_plots)

def untargeted_jsma(data_sample, model, i_max, eps, is_increasing=True, min=0.0, max=1.0, use_logits=False, show_plots=False):
    return jsma(data_sample, model, i_max, eps, min=min, max=max, is_increasing=is_increasing, use_logits=use_logits, show_plots=show_plots)

def test_jsma():
    # model = DenseModel().load_model_data()
    model = ConvModel().load_model_data()
    eval_dataset = model.get_dataset(tfds.Split.TEST, batch_size=1)
    for data_sample in eval_dataset.take(20):
        image, true_label = data_sample
        target_label = true_label.numpy().squeeze().argmax() - 2 % model.get_number_of_classes()
        if true_label.numpy().squeeze().argmax() != model(image).numpy().squeeze().argmax():
            continue
        show_plot(model(image), image, model.get_label_names())
        # ret_image = untargeted_jsma(data_sample, model, 50, 1.0, is_increasing=True, use_logits=False, show_plots=True)
        ret_image = targeted_jsma(data_sample, model, 50, 1.0, target_label, is_increasing=True, use_logits=True, show_plots=False)
        show_plot(model(ret_image), ret_image, model.get_label_names())

if __name__ == "__main__":
    enable_eager_execution()
    test_jsma()
