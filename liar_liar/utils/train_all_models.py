from liar_liar.models.cifar_100_models import CIFAR100ConvModel
from liar_liar.models.cifar_10_models import CIFAR10ConvModel
from liar_liar.models.mnist_models.let_net_5 import LeNet5
from liar_liar.models.mnist_models import MNISTConvModel
from liar_liar.models.mnist_models.mnist_dense_model import MNISTDenseModel

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