from liar_liar.base_models.sequential_model import SequentialModel


class CIFAR10Model(SequentialModel):

    def __init__(self, optimizer, loss, metrics, MODEL_NAME):
        super().__init__(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics,
                         MODEL_NAME=MODEL_NAME,
                         dataset_name='cifar10')


def get_cifar10_models():
    """
    Returns:
        list[CIFAR10Model]:
    """
    from liar_liar.cifar_10_models.cifar_10_conv_model import CIFAR10ConvModel
    from liar_liar.cifar_10_models.simplenet_cifar10 import SimpleNetCIFAR10
    return [
        # CIFAR10ConvModel().load_model_data(),
        SimpleNetCIFAR10().load_model_data()
    ]