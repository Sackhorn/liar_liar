import tensorflow as tf

def count(model, ret_image, labels):
    ret_labels = model(ret_image)
    ret_labels = tf.one_hot(tf.argmax(ret_labels, axis=1), model.get_number_of_classes())
    has_succeeded = tf.math.reduce_all(tf.math.equal(ret_labels, labels), axis=1)
    has_succeeded = list(has_succeeded.numpy())
    return has_succeeded.count(True)/len(has_succeeded)

def count_untargeted(model, ret_image, labels):
    ret_labels = model(ret_image)
    ret_labels = tf.one_hot(tf.argmax(ret_labels, axis=1), model.get_number_of_classes())
    has_succeeded = tf.math.reduce_all(tf.math.equal(ret_labels, labels), axis=1)
    has_succeeded = list(has_succeeded.numpy())
    return has_succeeded.count(False)/len(has_succeeded)