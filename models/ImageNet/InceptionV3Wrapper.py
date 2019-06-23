import tensorflow as tf
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
import tensorflow_datasets as tfds

from models.ImageNet.ImageNetModel import ImageNetModel


class ResNetWrapper(ImageNetModel):

    imagenet: ImageNetModel

    def __init__(self,
                 optimizer=Adam(),
                 loss=categorical_crossentropy,
                 metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="imagenet_v3")
        self.imagenet = InceptionV3()

    def call(self, input, get_raw=False, **kwargs):
        if get_raw:
            self.imagenet(input)
            return self.imagenet.layers[-1].output
        else:
            return self.imagenet(input)



if __name__ == "__main__":
    enable_eager_execution()
    model = ResNetWrapper()
    model.test()
