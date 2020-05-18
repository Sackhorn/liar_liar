import tensorflow as tf
# from tensorflow.python.keras.applications import ResNet152V2
from tensorflow.keras.applications import ResNet50V2
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import top_k_categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from liar_liar.base_models.model_names import RESNET_NAME
from liar_liar.image_net_models.image_net_model_base import ImageNetModel


class ResNetWrapper(ImageNetModel):

    resnet: ImageNetModel

    def __init__(self,
                 optimizer=Adam(),
                 loss=categorical_crossentropy,
                 metrics=[top_k_categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME=RESNET_NAME)
        self.resnet = ResNet50V2()
        self.resnet.layers[-1].activation = tf.keras.activations.linear

    @tf.function
    def call(self, input, get_raw=False, **kwargs):
        if get_raw:
            return self.resnet(input)
        else:
            return tf.nn.softmax(self.resnet(input))

    def load_model_data(self):
        print("calling load_model_data on ResNetWrapper is unnecessary as keras takes care of that for us")
        return self


if __name__ == "__main__":
    model = ResNetWrapper()
    model.test()
