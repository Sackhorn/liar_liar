from tensorflow.python.ops.losses.losses_impl import softmax_cross_entropy
from models.Attacks.attack import Attack
import tensorflow as tf

@tf.function
def _fgsm(data_sample, model, i_max=1, eps=0.35, target_label=None, min=0.0, max=1.0):
    image, label = data_sample
    eps = eps if target_label is None else -eps
    # We stack target_label for each element of batch, otherwise the shapes won't match
    label = label if target_label is None else tf.stack([target_label]*label.shape[0])
    for _ in tf.range(i_max):
        with tf.GradientTape() as tape:
            tape.watch(image)
            logits = model(image)
            loss = softmax_cross_entropy(label, logits)
        gradient = tape.gradient(loss, image)
        del tape
        image = image + eps * tf.sign(gradient)
        image = tf.clip_by_value(image, min, max)
    return image

class FGSM(Attack):
    @staticmethod
    def run_attack(model, data_sample, target_class):
        return_images = _fgsm(data_sample, model, i_max=500, eps=0.0002, target_label=target_class)
        return (return_images, model(return_images))