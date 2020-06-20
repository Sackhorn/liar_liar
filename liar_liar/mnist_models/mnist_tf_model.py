from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from liar_liar.base_models.model_names import MNIST_TF_NAME
from liar_liar.mnist_models.mnist_model_base import MNISTModel


# TODO: Add Exponential Decay to lr
# lr_scheduler = ExponentialDecay(0.05, decay_steps=100000, decay_rate=0.96)
class MNISTTFModel(MNISTModel):

    def __init__(self, optimizer=SGD(), loss=categorical_crossentropy, metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME=MNIST_TF_NAME)
        self.sequential_layers = [
            Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.4),
            Dense(10, activation='softmax')
        ]

if __name__ == "__main__":
    # model = MNISTTFModel()
    # lr_callback = ReduceLROnPlateau(factor=0.5, patience=2, verbose=1, min_delta=0.01)
    # model.train(epochs=50, batch_size=1024, augment_data=True, learning_rate_callback=lr_callback)
    model = MNISTTFModel().load_model_data()
    list = model.test()
    print(list)

