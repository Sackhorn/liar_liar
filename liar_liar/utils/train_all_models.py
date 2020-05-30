from liar_liar.cifar_100_models.cifar_100_conv_model import CIFAR100ConvModel
from liar_liar.cifar_10_models.cifar_10_conv_model import CIFAR10ConvModel
from liar_liar.mnist_models.let_net_5 import LeNet5
from liar_liar.mnist_models.mnist_conv_model import MNISTConvModel
from liar_liar.mnist_models.mnist_dense_model import MNISTDenseModel

model = CIFAR10ConvModel()
model.train(epochs=150)
model = CIFAR100ConvModel()
model.train(epochs=1000)
model = MNISTConvModel()
model.train(epochs=5)
model = MNISTDenseModel()
model.train(epochs=5)
model = LeNet5()
model.train(epochs=15, augment_data=False)