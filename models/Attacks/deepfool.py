#This is the implementation of method given in this paper
# Source https://arxiv.org/pdf/1511.04599.pdf
from models.BaseModels.SequentialModel import SequentialModel
import tensorflow as tf


def deepfool(classifier, data_sample, max_iter=10000, min=0.0, max=1.0):
    """

    Args:
        classifier: A classifier model that we want to attack
        data_sample: A tuple of tensors of structure (image_tensor, label_tensor)
        max_iter: Maximal number of iterations before returning
        min: Minimal value of input image
        max: Maximal value of input image

    Returns: A tuple of structure (adversarial_example, classifier output for examples)

    """
    return _deepfool(data_sample, classifier, max_iter=max_iter, min=min, max=max)

@tf.function
def _deepfool(data_sample, classifier, max_iter=10000, min=0.0, max=1.0):
    """

    :type classifier: SequentialModel
    """

    nmb_classes = classifier.get_number_of_classes()
    image, label = data_sample
    batch_size = image.shape[0]
    width = image[0].shape[0]
    height = image[0].shape[1]
    color_space = image[0].shape[2]
    iter = 0
    output = classifier(image)
    output = tf.one_hot(tf.argmax(output, axis=1), nmb_classes)
    is_properly_classified = tf.math.reduce_all(tf.math.equal(output, label), axis=1)


    while tf.math.reduce_any(is_properly_classified) and iter < max_iter:
        gradients_by_cls = tf.TensorArray(tf.float32, size=nmb_classes, element_shape=[batch_size, width, height, color_space])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(image)
            tmp_image = tf.transpose(image, perm=[1,0,2,3])
            tape.watch(tmp_image)
            logits = classifier(image)

            for k in tf.range(nmb_classes):
                tmp = logits[:, k]
                with tape.stop_recording():
                    grds = tape.gradient(tmp, image)
                gradients_by_cls  = gradients_by_cls.write(k, grds)
        del tape
        gradients_by_cls_tensor = tf.reshape(gradients_by_cls.concat(), [nmb_classes, batch_size, width, height, color_space])
        gradients_by_cls_tensor = tf.transpose(gradients_by_cls_tensor, perm=[1, 0, 2, 3, 4])
        classified_gradient = tf.gather_nd(gradients_by_cls_tensor, tf.reshape(tf.argmax(label, axis=1), [batch_size, 1]), batch_dims=1)
        classified_gradient = tf.reshape(tf.tile(classified_gradient, [1, nmb_classes, 1, 1]), [batch_size, nmb_classes, width, height, color_space])
        w_prime = gradients_by_cls_tensor - classified_gradient
        classified_logits = tf.tile(tf.reshape(tf.gather_nd(logits, tf.reshape(tf.argmax(label, axis=1), [batch_size, 1]), batch_dims=1), [batch_size, 1]),[1, nmb_classes])
        f_prime = logits - classified_logits
        coef_val = tf.abs(f_prime)/tf.norm(tf.norm(w_prime, axis=[-2, -1]), axis=2)
        coef_val_argmin = tf.math.argmin(coef_val, axis=1)
        coef_val_argmin = tf.reshape(coef_val_argmin, [batch_size, 1])
        w_l_prime = tf.gather_nd(w_prime, coef_val_argmin, batch_dims=1)
        f_l_prime = tf.gather_nd(f_prime, coef_val_argmin, batch_dims=1)
        scalar = tf.abs(f_l_prime)/tf.norm(tf.norm(w_l_prime, axis=[-2,-1]), axis=1)
        perturbation = tf.multiply(tf.reshape(scalar, [batch_size,1,1,1]),  w_l_prime)
        image = image + perturbation
        image = tf.clip_by_value(image, min, max)
        iter += 1
        output = classifier(image)
        output = tf.one_hot(tf.argmax(output, axis=1), nmb_classes)
        is_properly_classified = tf.math.reduce_all(tf.math.equal(output, label), axis=1)
    return (image, classifier(image))
