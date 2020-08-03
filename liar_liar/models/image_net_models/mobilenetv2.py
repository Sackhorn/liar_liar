import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from liar_liar.models.base_models.model_names import MOBILENETV2_NAME
from liar_liar.models.image_net_models.image_net_model_base import ImageNetModel


class MobileNetV2Wrapper(ImageNetModel):

    imagenet: MobileNetV2

    def __init__(self,optimizer=Adam(),
                 loss=categorical_crossentropy,
                 metrics=[categorical_accuracy, top_k_categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME=MOBILENETV2_NAME)
        self.mobilenetv2 = MobileNetV2()
        self.mobilenetv2.layers[-1].activation = tf.keras.activations.linear

    @tf.function
    def call(self, input, get_raw=False, **kwargs):
        if get_raw:
            return self.mobilenetv2(input)
        else:
            return tf.nn.softmax(self.mobilenetv2(input))

    def get_dataset(self, split, batch_size=32, shuffle=10000, filter=None, preprocess=None,
                    **kwargs) -> tf.data.Dataset:
        def preprocess_dataset(x, y):
            y = tf.one_hot(y, self.get_number_of_classes())
            x = tf.image.resize(x, (224, 224))
            x = tf.reshape(x, (224, 224, 3))
            x = preprocess_input(x)
            return x, y
        return super().get_dataset(split, batch_size, shuffle, filter, preprocess=preprocess_dataset, **kwargs)

    def load_model_data(self):
        print("calling load_model_data on MobileNetV2Wrapper is unnecessary as keras takes care of that for us")
        return self

    def get_input_shape(self):
        return (224, 224, 3)


if __name__ == "__main__":
    model = MobileNetV2Wrapper()
    model.test()
