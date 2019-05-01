from tensorflow.python import enable_eager_execution, Tensor
from models.MNISTModels.ConvModel import ConvModel
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt



def show_plot(logits, image):
    """

    :type image: Tensor
    """
    probs = tf.nn.softmax(logits)
    probs = probs.numpy().reshape(10).tolist()
    fig = plt.figure()
    img_plt = fig.add_subplot(121)
    img_plt.imshow(image.numpy().reshape(28, 28).astype(np.float32), cmap=plt.get_cmap("gray"))
    bar_plt = fig.add_subplot(122)
    bar_plt.bar(np.arange(10), probs)
    bar_plt.set_xticks(np.arange(10))
    plt.show()

def jsma(data_sample, model, i, eps, y_target=None, min=0.0, max=1.0):
    image, label = data_sample
    show_plot(model(image), image)
    all_pixels = generate_all_pixels(tf.squeeze(image).shape)
    iter = 0
    while iter < i and len(all_pixels) > 0:
        all_pixel_pairs = generate_all_pixel_pairs(all_pixels)
        chosen_pixel_pair = saliency_map(model, image, label, all_pixel_pairs)
        add_tensor = np.zeros((28,28))
        for pixel in chosen_pixel_pair:
            add_tensor[pixel[0]][pixel[1]] = 0.6
            pixel_val = image.numpy()[0][pixel[0]][pixel[1]]
            if pixel_val < 1e-10 or pixel_val > 0.99:
                all_pixels.remove(pixel)
        add_tensor = add_tensor.reshape((1,28,28,1))
        add_tensor = tf.convert_to_tensor(add_tensor, dtype=tf.float32)
        tf.reshape(add_tensor, (1,28,28,1))
        image = image + add_tensor
        image = tf.clip_by_value(image, min, max)
        show_plot(model(image), image)

        iter += 1
    show_plot(model(image), image)

def saliency_map(model, image, label, all_pixel_pairs):
    target_label = 9
    # true_label = tf.argmax(tf.squeeze(label)).numpy()
    grds_by_cls = []
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(image)
        logits = model(image)
        for k in range(10):
            grds_by_cls.append(tf.squeeze(tape.gradient(logits[0][k], image)).numpy())
    max = float("-inf")
    return_pixel_pair = None
    for pair in all_pixel_pairs:
        alpha = 0.0
        beta = 0.0
        for pixel in pair:
            alpha += grds_by_cls[target_label][pixel[0]][pixel[1]]
            for class_nmb in range(10):
                if class_nmb != target_label:
                    beta += grds_by_cls[class_nmb][pixel[0]][pixel[1]]
        if alpha > 0.0 and beta < 0 and -alpha*beta > max:
            return_pixel_pair = pair
            max = -alpha*beta
    return return_pixel_pair



def generate_all_pixels(shape):
    x, y = shape
    all_pixels = []
    for i in range(x):
        for j in range(y):
            all_pixels.append((i, j))
    return all_pixels

def generate_all_pixel_pairs(all_pixels):
    all_pairs = []
    for i in range(len(all_pixels)):
        first = all_pixels[i]
        for j in all_pixels[i+1:]:
            second = j
            all_pairs.append((first, second))
    return all_pairs


def execute_jsma(data_sample, model, i, eps, min=0.0, max=1.0):
    return jsma(data_sample, model, i, eps, min=min, max=max)

def test_fgsm_mnist():
    model = ConvModel()
    model.load_model_data()
    eval_dataset = model.get_dataset(tfds.Split.TEST, batch_size=1)
    for data_sample in eval_dataset.take(1):
        execute_jsma(data_sample, model, 80, 0.0001)

if __name__ == "__main__":
    enable_eager_execution()
    test_fgsm_mnist()
