from tensorflow.python import enable_eager_execution, Tensor
from models.CIFAR10Models.ConvModel import ConvModel
from models.MNISTModels.DenseModel import DenseModel
from models.utils.images import show_plot
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# TODO: Generalize for all models
# Source https://arxiv.org/pdf/1511.07528.pdf


def jsma(data_sample, model, i, eps, target_label, min=0.0, max=1.0):
    image, true_label = data_sample

    true_label = true_label.numpy().squeeze().argmax()
    current_prediction = true_label

    show_plot(model(image), image, model.get_label_names())
    input_shape = model.get_input_shape()

    all_pixels = generate_all_pixels(input_shape)

    iter = 0
    while iter < i and len(all_pixels) > 0 and current_prediction != target_label:
        all_pixels_pairs = generate_all_pixel_pairs(all_pixels)
        chosen_pixel_pair = saliency_map(model, image, true_label, target_label, all_pixels_pairs)
        add_tensor = np.zeros(input_shape)
        for pixel in chosen_pixel_pair:
            if input_shape[2] > 1:
                add_tensor[pixel[0]][pixel[1]][pixel[2]] = eps
                pixel_val = image.numpy()[0][pixel[0]][pixel[1]][pixel[2]] + eps
            else:
                add_tensor[pixel[0]][pixel[1]] = eps
                pixel_val = image.numpy()[0][pixel[0]][pixel[1]] + eps

            if pixel_val < 1e-10 or pixel_val > 0.99:
                all_pixels.remove(pixel)

        add_tensor = add_tensor.reshape(input_shape)
        add_tensor = tf.convert_to_tensor(add_tensor, dtype=tf.float32)
        tf.reshape(add_tensor, input_shape)
        image = image + add_tensor
        image = tf.clip_by_value(image, min, max)
        current_prediction = model(image)
        # show_plot(current_prediction, image, model.get_label_names())
        current_prediction = current_prediction.numpy().squeeze().argmax()
        iter += 1

    show_plot(model(image), image, model.get_label_names())

# TODO:nie traktować każdej barwy jako osobny ficzer tylko sumować i zmieniać intensywność
def saliency_map(model, image, true_label, target_label, all_pixel_pairs):
    nmb_classes = model.get_number_of_classes()
    input_shape = model.get_input_shape()
    grds_by_cls = []
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(image)
        logits = model(image)
        for k in range(nmb_classes):
            grds_by_cls.append(tf.squeeze(tape.gradient(logits[0][k], image)).numpy())

    max = float("-inf")
    return_pixel_pair = None


    for pixel_pair in all_pixel_pairs:
        alpha = 0.0
        beta = 0.0
        first, second = pixel_pair

        if input_shape[2] > 1:
            alpha += grds_by_cls[target_label][first[0]][first[1]][first[2]]
            alpha += grds_by_cls[target_label][second[0]][second[1]][second[2]]
        else:
            alpha += grds_by_cls[target_label][first[0]][first[1]]
            alpha += grds_by_cls[target_label][second[0]][second[1]]

        for class_nmb in range(nmb_classes):
            if class_nmb != target_label:
                if input_shape[2] > 1:
                    beta += grds_by_cls[class_nmb][first[0]][first[1]][first[2]]
                    beta += grds_by_cls[class_nmb][second[0]][second[1]][second[2]]
                else:
                    beta += grds_by_cls[class_nmb][first[0]][first[1]]
                    beta += grds_by_cls[class_nmb][second[0]][second[1]]

        if alpha > 0.0 and beta < 0 and -alpha*beta > max:
            return_pixel_pair = pixel_pair
            max = -alpha*beta

    return return_pixel_pair



def generate_all_pixels(shape):
    x, y, z = shape
    all_pixels = []
    for i in range(x):
        for j in range(y):
            for k in range(z):
                all_pixels.append((i, j, k))
    return all_pixels

def generate_all_pixel_pairs(all_pixels):
    all_pairs = []
    for i in range(len(all_pixels)):
        first = all_pixels[i]
        for j in all_pixels[i+1:]:
            second = j
            all_pairs.append((first, second))
    return all_pairs


def execute_jsma(data_sample, model, i, eps, target_label,  min=0.0, max=1.0):
    return jsma(data_sample, model, i, eps, target_label, min=min, max=max)

def test_fgsm_mnist():
    # model = ConvModel().load_model_data()
    model = DenseModel().load_model_data()
    eval_dataset = model.get_dataset(tfds.Split.TEST, batch_size=1)
    for data_sample in eval_dataset.take(5):
        _, true_label = data_sample
        target_label = true_label.numpy().squeeze().argmax() + 1 % model.get_number_of_classes()
        execute_jsma(data_sample, model, 50, 1.0, target_label)

if __name__ == "__main__":
    enable_eager_execution()
    test_fgsm_mnist()
