import tensorflow as tf
from tensorflow.python.keras.initializers import glorot_normal, RandomNormal
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.metrics import categorical_accuracy

from liar_liar.base_models.model_names import SIMPLENET_CIFAR10_NAME
from liar_liar.cifar_10_models.cifar_10_model_base import CIFAR10Model

LEARNING_RATE = 0.001  #values between 0.001 and 0.00001
WEIGHT_DECAY = 0.005  #L2 regularization

# if(EPOCH == 0):  #dropout at training time only
#     conv_dropout = 0
# else:
#     conv_dropout = 0.2  #between 0.1 and 0.3, batch normalization reduces the needs of dropout
conv_dropout = 0.1  #between 0.1 and 0.3, batch normalization reduces the needs of dropout

convStrides = 1  # stride 1 allows us to leave all spatial down-sampling to the POOL layers
poolStrides = 2

convKernelSize = 3
convKernelSize1 = 1
poolKernelSize = 2

filterSize1 = 64
filterSize = 128

bn_decay = 0.95
batch_epsilon = 1e-5
act = tf.keras.activations.relu
s = 2

class SimpleNetCIFAR10(CIFAR10Model):

    def __init__(self, optimizer=tf.keras.optimizers.Adadelta(lr=0.9, rho=0.9), loss=tf.keras.losses.categorical_crossentropy, metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME=SIMPLENET_CIFAR10_NAME)
        self.sequential_layers = [
            # Block 1
            Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_normal()),
            BatchNormalization(),
            Activation(act),
            Dropout(0.2),

            # Block 2
            Conv2D(128, (3, 3), padding='same', kernel_initializer=glorot_normal()),
            BatchNormalization(),
            Activation(act),
            Dropout(0.2),

            # Block 3
            Conv2D(128, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)),
            BatchNormalization(),
            Activation(act),
            Dropout(0.2),

            # Block 4
            Conv2D(128, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)),
            BatchNormalization(),
            Activation(act),
            # First Maxpooling
            MaxPooling2D(pool_size=(2, 2), strides=s),
            Dropout(0.2),


            # Block 5
            Conv2D(128, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)),
            BatchNormalization(),
            Activation(act),
            Dropout(0.2),

            # Block 6
            Conv2D(128, (3, 3), padding='same', kernel_initializer=glorot_normal()),
            BatchNormalization(),
            Activation(act),
            Dropout(0.2),

            # Block 7
            Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_normal()),
            # Second Maxpooling
            MaxPooling2D(pool_size=(2, 2), strides=s),
            BatchNormalization(),
            Activation(act),
            Dropout(0.2),


            # Block 8
            Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_normal()),
            BatchNormalization(),
            Activation(act),
            Dropout(0.2),

            # Block 9
            Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_normal()),
            BatchNormalization(),
            Activation(act),
            Dropout(0.2),
            # Third Maxpooling
            MaxPooling2D(pool_size=(2, 2), strides=s),


            # Block 10
            Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_normal()),
            BatchNormalization(),
            Activation(act),
            Dropout(0.2),

            # Block 11  
            Conv2D(2048, (1, 1), padding='same', kernel_initializer=glorot_normal()),
            Activation(act),
            Dropout(0.2),

            # Block 12  
            Conv2D(256, (1, 1), padding='same', kernel_initializer=glorot_normal()),
            Activation(act),
            # Fourth Maxpooling
            MaxPooling2D(pool_size=(2, 2), strides=s),
            Dropout(0.2),


            # Block 13
            Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_normal()),
            Activation(act),
            # Fifth Maxpooling
            MaxPooling2D(pool_size=(2, 2), strides=s),

            # Final Classifier
            Flatten(),
            Dense(self.get_number_of_classes(), activation='softmax'),
]

if __name__ == "__main__":
    # model = SimpleNetCIFAR10()
    # model.train(epochs=200, batch_size=256)
    model = SimpleNetCIFAR10().load_model_data()
    model.test()