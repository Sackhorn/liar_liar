from liar_liar.base_models.sequential_model import SequentialModel



class MNISTModel(SequentialModel):

    def __init__(self, optimizer, loss, metrics, MODEL_NAME):
        super().__init__(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics,
                         MODEL_NAME=MODEL_NAME,
                         dataset_name='mnist')


def get_mnist_models():
    """
    Returns:
        list[MNISTModel]:
    """
    from liar_liar.mnist_models.let_net_5 import LeNet5
    from liar_liar.mnist_models.mnist_conv_model import MNISTConvModel
    from liar_liar.mnist_models.mnist_tf_model import MNISTTFModel
    # from liar_liar.mnist_models.mnist_dense_model import MNISTDenseModel
    return [
        # MNISTConvModel().load_model_data(),
        MNISTTFModel().load_model_data(),
        # MNISTDenseModel().load_model_data(),
        LeNet5().load_model_data()
    ]
