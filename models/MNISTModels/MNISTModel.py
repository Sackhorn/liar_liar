from models.BaseModels.SequentialModel import SequentialModel


class MNISTModel(SequentialModel):

    def __init__(self, optimizer, loss, metrics, MODEL_NAME):
        super().__init__(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics,
                         MODEL_NAME=MODEL_NAME,
                         dataset_name='mnist')
