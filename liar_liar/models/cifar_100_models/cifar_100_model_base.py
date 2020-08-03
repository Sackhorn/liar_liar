from liar_liar.models.base_models.sequential_model import SequentialModel


class CIFAR100Model(SequentialModel):

    def __init__(self, optimizer, loss, metrics, MODEL_NAME):
        super().__init__(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics,
                         MODEL_NAME=MODEL_NAME,
                         dataset_name='cifar100')


def get_cifar100_models():
    """
    Returns:
        list[CIFAR100Model]:
    """
    from liar_liar.models.cifar_100_models.simplenet_cifar100 import SimpleNetCIFAR100
    return [
        # CIFAR100ConvModel().load_model_data(),
        SimpleNetCIFAR100().load_model_data()
    ]