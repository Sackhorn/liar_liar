#This is the implementation of method given in this paper
# https://arxiv.org/pdf/1412.6572.pdf
from tensorflow.python.ops.losses.losses_impl import softmax_cross_entropy
import tensorflow as tf

def fgsm(classifier, data_sample, target_class=None, iter_max=1, eps=0.01, min=0.0, max=1.0):
    """

    Args:
        classifier: A classifier model that we want to attack
        data_sample: A tuple of tensors of structure (image_tensor, label_tensor) against wich attack is run
        target_class: When passed None function performs an untargeted attack, otherwise it targets the class
            encoded in one hot label.
        iter_max: Number of times the image is update according to FGSM method.
        eps: An amount by wich input image will be modified in each step of the iteration
        min: Minimal value of input image
        max: Maximal value of input image

    Returns: A tuple of structure (adversarial_example, classifier output for examples)

    """
    return_images = _fgsm(data_sample, classifier, target_class, iter_max, eps, min, max)
    parameters = {
        "target_class" : target_class is not None,
        "iter_max" : iter_max,
        "eps" : eps,
        # "min" : min,
        # "max" : max
    }
    return (return_images, classifier(return_images), parameters)

def fgsm_untargeted_wrapper(iter_max=1, eps=0.01, min=0.0, max=1.0):
    """
    This wraps FGSM call in a handy way that allows us using this as unspecified untargeted attack method
    Returns: Wrapped FGSM for untargeted attack format

    """
    def wrapped_fgsm(classifier, data_sample):
        return fgsm(classifier, data_sample, None, iter_max, eps, min, max)
    return wrapped_fgsm


def fgsm_targeted_wrapper(iter_max=1, eps=0.01, min=0.0, max=1.0):
    """
    This wraps FGSM call in a handy way that allows us using this as unspecified targeted attack method
    Returns: Wrapped FGSM for targeted attack format

    """
    def wrapped_fgsm(classifier, data_sample, target_class):
        return fgsm(classifier, data_sample, target_class, iter_max, eps, min, max)
    return wrapped_fgsm


@tf.function
def _fgsm(data_sample, classifier, target_class=None, iter_max=1, eps=0.35, min=0.0, max=1.0):
    image, label = data_sample
    eps = eps if target_class is None else -eps
    # We stack target_label for each element of batch, otherwise the shapes won't match
    label = label if target_class is None else tf.stack([target_class] * label.shape[0])
    for _ in tf.range(iter_max):
        with tf.GradientTape() as tape:
            tape.watch(image)
            logits = classifier(image)
            loss = softmax_cross_entropy(label, logits)
        gradient = tape.gradient(loss, image)
        del tape
        image = image + eps * tf.sign(gradient)
        image = tf.clip_by_value(image, min, max)
    return image
