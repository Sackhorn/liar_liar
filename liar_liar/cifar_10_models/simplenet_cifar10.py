import tensorflow as tf
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizers import Adadelta

from liar_liar.base_models.model_names import SIMPLENET_CIFAR10_NAME
from liar_liar.cifar_10_models.cifar_10_model_base import CIFAR10Model

LEARNING_RATE = 0.001  #values between 0.001 and 0.00001
WEIGHT_DECAY = 0.005  #L2 regularization

# if(EPOCH == 0):  #dropout at training time only
#     conv_dropout = 0
# else:
#     conv_dropout = 0.2  #between 0.1 and 0.3, batch normalization reduces the needs of dropout
conv_dropout = 0.2  #between 0.1 and 0.3, batch normalization reduces the needs of dropout

convStrides = 1  # stride 1 allows us to leave all spatial down-sampling to the POOL layers
poolStrides = 2

convKernelSize = 3
convKernelSize1 = 1
poolKernelSize = 2

filterSize1 = 64
filterSize = 128

bn_decay = 0.95



class SimpleNetCIFAR10(CIFAR10Model):

    def __init__(self, optimizer=tf.keras.optimizers.Adadelta(lr=0.1, rho=0.9, epsilon=1e-3, decay=0.001), loss=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2, metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME=SIMPLENET_CIFAR10_NAME)
        self.sequential_layers = [
tf.keras.layers.Conv2D( filterSize1, kernel_size=[convKernelSize, convKernelSize],
                                     strides=(convStrides, convStrides), padding="SAME",
                                     kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                     bias_initializer=tf.constant_initializer(0.1),
                                     kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv1'),
tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
tf.keras.layers.LeakyReLU(alpha=0.1, name='act1'),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize],
                                 strides=(convStrides, convStrides), padding="SAME",
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv2'),
tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
tf.keras.layers.MaxPool2D(strides=(poolStrides, poolStrides), padding="SAME"),
tf.keras.layers.LeakyReLU(alpha=0.1, name='act2'),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize],
                                 strides=(convStrides, convStrides), padding="SAME",
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv3'),
tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
tf.keras.layers.MaxPool2D(strides=(poolStrides, poolStrides), padding="SAME"),
tf.keras.layers.LeakyReLU(alpha=0.1, name='act3'),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize],
                                 strides=(convStrides, convStrides), padding="SAME",
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv4'),
tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
tf.keras.layers.MaxPool2D(strides=(poolStrides, poolStrides), padding="SAME"),
tf.keras.layers.LeakyReLU(alpha=0.1, name='act4'),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize],
                                 strides=(convStrides, convStrides), padding="SAME",
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv5'),
tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
tf.keras.layers.LeakyReLU(alpha=0.1, name='act5'),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize],
                                 strides=(convStrides, convStrides), padding="SAME",
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv6'),
tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
tf.keras.layers.LeakyReLU(alpha=0.1, name='act6'),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize],
                                 strides=(convStrides, convStrides), padding="SAME",
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv7'),
tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
tf.keras.layers.MaxPool2D(strides=(poolStrides, poolStrides), padding="SAME"),
tf.keras.layers.LeakyReLU(alpha=0.1, name='act7'),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize],
                                 strides=(convStrides, convStrides), padding="SAME",
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv8'),
tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
tf.keras.layers.MaxPool2D(strides=(poolStrides, poolStrides), padding="SAME"),
tf.keras.layers.LeakyReLU(alpha=0.1, name='act8'),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize],
                                 strides=(convStrides, convStrides), padding="SAME",
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv9'),
tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
tf.keras.layers.MaxPool2D(strides=(poolStrides, poolStrides), padding="SAME"),
tf.keras.layers.LeakyReLU(alpha=0.1, name='act9'),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.Conv2D(filterSize, kernel_size=[convKernelSize, convKernelSize],
                                  strides=(convStrides, convStrides), padding="SAME",
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv10'),
tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
tf.keras.layers.LeakyReLU(alpha=0.1, name='act10'),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.Conv2D(filterSize, kernel_size=[convKernelSize1, convKernelSize1],
                                  strides=(convStrides, convStrides), padding="SAME",
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv11'),
tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
tf.keras.layers.LeakyReLU(alpha=0.1, name='act11'),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.Conv2D(filterSize, kernel_size=[convKernelSize1, convKernelSize1],
                                  strides=(convStrides, convStrides), padding="SAME",
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv12'),
tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
tf.keras.layers.LeakyReLU(alpha=0.1, name='act12'),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.Conv2D(self.get_number_of_classes(), kernel_size=[convKernelSize, convKernelSize],
                                  strides=(convStrides, convStrides), padding="SAME",
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name='conv13'),
tf.keras.layers.MaxPool2D(strides=(poolStrides, poolStrides), padding="SAME"),
tf.keras.layers.Dropout(rate=conv_dropout),
tf.keras.layers.GlobalAveragePooling2D(name='avg13')]

if __name__ == "__main__":
    model = SimpleNetCIFAR10()
    model.train(epochs=1000)
