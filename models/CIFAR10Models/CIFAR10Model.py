from models.BaseModels.SequentialModel import SequentialModel


class CIFAR10Model(SequentialModel):

    def __init__(self, optimizer, loss, metrics, MODEL_NAME=""):
        super().__init__(nmb_classes=10,
                         optimizer=optimizer,
                         loss=loss,
                         metrics=metrics,
                         MODEL_NAME=MODEL_NAME)

    def get_dataset(self, split, name='cifar10', batch_size=32, shuffle=10000, nmb_classes=10):
        return super().get_dataset(split, name, batch_size, shuffle, nmb_classes)