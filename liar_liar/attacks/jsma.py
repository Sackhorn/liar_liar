#This is the implementation of method given in this paper
# Source https://arxiv.org/pdf/1511.07528.pdf
import time
from math import floor

import numpy as np
import tensorflow as tf

from liar_liar.attacks.attack import Attack


class JSMATargeted(Attack):

    def init_wrapper(self, *args, **kwargs):
        self._wrapper = jsma_targeted_wrapper(*args, **kwargs)

class JSMAUntargeted(Attack):

    def init_wrapper(self, *args, **kwargs):
        self._wrapper = jsma_untargeted_wrapper(*args, **kwargs)

def jsma(classifier,
         data_sample,
         target_class=None,
         max_perturbation=0.1,
         theta=1,
         is_increasing=True,
         min=0.0,
         max=1.0,
         show_plots=False,
         use_logits=False):
    image, label = data_sample
    arr_image = []
    for i in range(len(image)):
        ret_image, _ = _jsma((tf.expand_dims(image[i], 0), label[i]),
                             classifier,
                             max_perturbation,
                             theta,
                             target_class,
                             is_increasing,
                             min,
                             max,
                             show_plots,
                             use_logits)
        arr_image.append(ret_image)
    arr_image = tf.concat(arr_image, 0)
    parameters = {
        # "target_class": target_class is not None,
        "max_perturbation":max_perturbation,
        "theta":theta,
        "is_increasing":is_increasing,
        "use_logits":use_logits,
        # "min":min,
        # "max":max
    }
    return (arr_image, classifier(arr_image), parameters)

def jsma_untargeted_wrapper(max_perturbation=0.1,
                            theta=1,
                            is_increasing=True,
                            min=0.0,
                            max=1.0,
                            show_plots=False,
                            use_logits=False):
    def wrapped_jsma(classifier, data_sample):
        return jsma(classifier,
                    data_sample,
                    max_perturbation=max_perturbation,
                    theta=theta,
                    is_increasing=is_increasing,
                    min=min,
                    max=max,
                    show_plots=show_plots,
                    use_logits=use_logits)
    return wrapped_jsma

def jsma_targeted_wrapper(max_perturbation=0.1,
                          theta=1,
                          is_increasing=True,
                          min=0.0,
                          max=1.0,
                          show_plots=False,
                          use_logits=False):
    def wrapped_jsma(classifier, data_sample, target_class):
        return jsma(classifier,
                    data_sample,
                    target_class=target_class,
                    max_perturbation=max_perturbation,
                    theta=theta,
                    is_increasing=is_increasing,
                    min=min,
                    max=max,
                    show_plots=show_plots,
                    use_logits=use_logits)
    return wrapped_jsma

def _jsma(data_sample,
         model,
         max_perturbation,
         theta=1,
         target_label=None,
         is_increasing=True,
         min=0.0,
         max=1.0,
         show_plots=False,
         use_logits=False):

    target_label = target_label.numpy().squeeze().argmax() if target_label is not None else None
    is_targeted = target_label is not None
    image, true_label = data_sample
    true_label = true_label.numpy().squeeze().argmax()
    current_prediction = model(image).numpy().squeeze().argmax()
    # if true_label != current_prediction:
    #     raise ValueError("The image given was wrongly classified in the first place")

    input_shape = model.get_input_shape()
    i_max = floor(np.prod(input_shape) * max_perturbation)
    all_pixels = generate_all_pixels(input_shape)
    all_pixels = prune_saturated_pixels(all_pixels, image, is_increasing, min, max)
    iter = 0

    # if show_plots:
    #     show_plot(model(image), image, labels_names=model.get_label_names())

    # this function determines whether the algorithm should end work
    def has_met_target(is_targeted, target_label, true_label, current_prediction):
        if is_targeted:
            return current_prediction == target_label
        else:
            return current_prediction != true_label

    while iter < i_max and len(all_pixels) > 0 and not has_met_target(is_targeted, target_label, true_label, current_prediction):
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
        # if show_plots:
        #     show_plot(model(image), image, labels_names=model.get_label_names())


    return image, current_prediction


def saliency_map(model, image, true_label, target_label, all_pixels, is_targeted, is_increasing, use_logits):


    nmb_classes = model.get_number_of_classes()


    # Generate forward derivatives for each of the classes
    grds_by_cls = []
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(image)
        logits = model(image, get_raw=use_logits)
        for k in range(nmb_classes):
            #TODO: Add stop watch in tape ang get results in TensorArray
            grds_by_cls.append(tape.gradient(logits[0][k], image).numpy().squeeze())

    grds_by_cls = np.array(grds_by_cls)
    first_pix, second_pix = generate_test(all_pixels)

    # This is important in case of un-targeted attack when we want to consider all classes except the true one
    # not just the target_class
    classes_to_iterate_over = [target_label] if is_targeted else list(range(nmb_classes))
    max_value = 0.0
    best_pair = None

    # for target_class in classes_to_iterate_over:

    #Here we differentiate between images with gray scale and images with color
    # This is a dirty hack we should get rid of it
    if tf.shape(tf.shape(tf.squeeze(image))).numpy() > 2:
        alpha = grds_by_cls[target_label, first_pix[:, 0], first_pix[:, 1], first_pix[:, 2]] + grds_by_cls[target_label, second_pix[:, 0], second_pix[:, 1], second_pix[:, 2]]
        beta = grds_by_cls[:, first_pix[:, 0], first_pix[:, 1], first_pix[:, 2]] + grds_by_cls[:, second_pix[:, 0], second_pix[:, 1], second_pix[:, 2]]
    else:
        alpha = grds_by_cls[target_label, first_pix[:, 0], first_pix[:, 1]] + grds_by_cls[target_label, second_pix[:, 0], second_pix[:, 1]]
        beta = grds_by_cls[:, first_pix[:, 0], first_pix[:, 1]] + grds_by_cls[:, second_pix[:, 0], second_pix[:, 1]]
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
    return first, second
#
# class jsma_plus_increasing(Attack):
#     @staticmethod
#     def run_attack(model, data_sample, target_class=None):
#         return _jsma(data_sample, model, 0.14, target_label=target_class)
#
# class jsma_plus_increasing_logits(Attack):
#     @staticmethod
#     def run_attack(model, data_sample, target_class=None):
#         return _jsma(data_sample, model, 0.14, target_label=target_class, use_logits=True)
#
# class jsma_minus_increasing(Attack):
#     @staticmethod
#     def run_attack(model, data_sample, target_class=None):
#         return _jsma(data_sample, model, 0.14, target_label=target_class, is_increasing=False)
#
# class jsma_minus_increasing_logits(Attack):
#     @staticmethod
#     def run_attack(model, data_sample, target_class=None):
#         return _jsma(data_sample, model, 0.14, target_label=target_class, is_increasing=False, use_logits=True)



# def test_jsma():
#     model = ConvModel().load_model_data()
#     eval_dataset = model.get_dataset(tfds.Split.TEST, batch_size=1)
#     for data_sample in eval_dataset.take(20):
#         image, true_label = data_sample
#         target_label = true_label.numpy().squeeze().argmax() - 2 % model.get_number_of_classes()
#         if true_label.numpy().squeeze().argmax() != model(image).numpy().squeeze().argmax():
#             continue
#
#         show_plot(model(image), image, model.get_label_names())
#         ret_image, ret_label = jsma_plus_increasing_logits.run_attack(model, data_sample, target_label)
#         show_plot(model(ret_image), ret_image, model.get_label_names())
#
# if __name__ == "__main__":
#     enable_eager_execution()
#     test_jsma()
