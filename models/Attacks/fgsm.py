from tensorflow.python.ops.losses.losses_impl import softmax_cross_entropy
import tensorflow as tf

def fgsm(classifier, data_sample, target_class=None, i_max=1, eps=0.01, min=0.0, max=1.0):
    """

    Args:
        classifier: A classifier model that we want to attack
        data_sample: A tuple of tensors of structure (image_tensor, label_tensor) against wich attack is run
        target_class: When passed None function performs an untargeted attack, otherwise it targets the class
            encoded in one hot label.
        i_max: Number of times the image is update according to FGSM method.
        eps: An amount by wich input image will be modified in each step of the iteration
        min: Minimal value of input image
        max: Maximal value of input image

    Returns: A tuple of structure (adversarial_example, classifier output for examples)

    """
    return_images = _fgsm(data_sample, classifier, i_max, eps, target_class, min, max)
    return (return_images, classifier(return_images))

@tf.function
def _fgsm(data_sample, classifier, i_max=1, eps=0.35, target_label=None, min=0.0, max=1.0):
    image, label = data_sample
    eps = eps if target_label is None else -eps
    # We stack target_label for each element of batch, otherwise the shapes won't match
    label = label if target_label is None else tf.stack([target_label]*label.shape[0])
    for _ in tf.range(i_max):
        with tf.GradientTape() as tape:
            tape.watch(image)
            logits = classifier(image)
            loss = softmax_cross_entropy(label, logits)
        gradient = tape.gradient(loss, image)
        del tape
        image = image + eps * tf.sign(gradient)
        image = tf.clip_by_value(image, min, max)
    return image
