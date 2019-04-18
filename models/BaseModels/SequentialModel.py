from models.BaseModels.ModelBase import ModelBase


class SequentialModel(ModelBase):

    def __init__(self, MODEL_NAME=""):
        super(SequentialModel, self).__init__(MODEL_NAME=MODEL_NAME)

    def call(self, input):
        result = self.sequential_layers[0](input)
        for layer in self.sequential_layers[1:]:
            result = layer(result)
        return result

