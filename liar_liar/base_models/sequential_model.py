from typing import List

from tensorflow.python import keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow_datasets import Split

from liar_liar.base_models.data_provider import DataProvider
from liar_liar.utils.utils import get_gcs_path


class SequentialModel(Model, DataProvider):

    sequential_layers: List[keras.layers.Layer]

    def __init__(self, optimizer, loss, metrics, MODEL_NAME="", dataset_name='', dataset_dir=None):
        dataset_dir = get_gcs_path() if get_gcs_path() is not None else dataset_dir
        super(SequentialModel, self).__init__()
        self.register_data_provider(MODEL_NAME, dataset_name, dataset_dir)
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.sequential_layers = []

    @tf.function
    def call(self, input, training=False, get_raw=False, **kwargs):
        result = input
        for layer in self.sequential_layers[:-1]:
            result = layer(result, training=training)
        if get_raw:
            activation_function = self.sequential_layers[-1].activation
            self.sequential_layers[-1].activation = lambda x: x
            result = self.sequential_layers[-1](result)
            self.sequential_layers[-1].activation = activation_function
        else:
            result = self.sequential_layers[-1](result)
        return result

    def train(self,
              epochs=1,
              train_data=None,
              callbacks=[],
              augment_data=True,
              batch_size=32,
              monitor="val_categorical_accuracy",
              learning_rate_callback=None):
        tsboard = TensorBoard(log_dir=self.get_tensorboard_path(), histogram_freq=10, write_graph=True, write_images=True)
        checkpoint = ModelCheckpoint(self.SAVE_DIR, save_best_only=True, save_weights_only=True, monitor=monitor)
        lr_callback = ReduceLROnPlateau(factor=0.75, patience=5, verbose=1, min_delta=0.005) if learning_rate_callback is None else learning_rate_callback
        callback = [tsboard, lr_callback, checkpoint] + callbacks

        test = self.get_dataset(Split.TEST)
        train = self.get_dataset(Split.TRAIN, augment_data=augment_data, batch_size=batch_size) if train_data is None else train_data
        self.fit(train.repeat(),
                 epochs=epochs,
                 steps_per_epoch=self.train_steps_per_epoch,
                 callbacks=callback,
                 validation_data=test,
                 validation_steps=self.test_steps)

        self.evaluate(test, steps=self.test_steps)
        self.save_model_data()

    def test(self, test_data=None):
        test = self.get_dataset(Split.TEST) if test_data is None else test_data
        self.evaluate(test, steps=self.test_steps)


def get_all_models():
    """
    Returns:
        list[SequentialModel]:
    """
    from liar_liar.mnist_models.mnist_model_base import get_mnist_models
    from liar_liar.cifar_100_models.cifar_100_model_base import get_cifar100_models
    from liar_liar.image_net_models.image_net_model_base import get_imagenet_models
    from liar_liar.cifar_10_models.cifar_10_model_base import get_cifar10_models
    models = []
    models.extend(get_mnist_models())
    models.extend(get_cifar10_models())
    models.extend(get_cifar100_models())
    models.extend(get_imagenet_models())
    return models