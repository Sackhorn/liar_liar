import tensorflow as tf
# from tensorflow.python.keras.applications import InceptionV3
from tensorflow.keras.applications import InceptionV3
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import top_k_categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from liar_liar.base_models.model_names import INCEPTION_V3_NAME
from liar_liar.image_net_models.image_net_model_base import ImageNetModel


class InceptionV3Wrapper(ImageNetModel):

    imagenet: ImageNetModel

    def __init__(self,
                 optimizer=Adam(),
                 loss=categorical_crossentropy,
                 metrics=[top_k_categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME=INCEPTION_V3_NAME)
        self.imagenet = InceptionV3()
        self.imagenet.layers[-1].activation = tf.keras.activations.linear

    @tf.function
    def call(self, input, get_raw=False, **kwargs):
        if get_raw:
            return self.imagenet(input)
        else:
            return tf.nn.softmax(self.imagenet(input))

    def load_model_data(self):
        print("calling load_model_data on InceptionV3Wrapper is unnecessary as keras takes care of that for us")
        return self


if __name__ == "__main__":
    model = InceptionV3Wrapper()
    model.test()
