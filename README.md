Liar Liar - Adversarial Attacks Testing Suite
=============================================
Liar Liar is a library based on Tensorflow 2 and Keras implementing multiple known adversarial attacks.
It's goal is to provide a simple API for testing models resistance and for creating samples for adversarial training.

Requirements
______________
- Python 3.x

Installation
-------------
```bash
    git clone http://github.com/Sackhorn/liar_liar
    cd liar_liar
    pip install -r requirements.txt
```

Usage
------------------

First you need to overload the `SequentialModel` if you want to use
a dataset that's not supported by the library but is available in tensorflow_datasets.
Then create a class that defines your model
```python
import tensorflow_datasets as tfds
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, AveragePooling2D
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from liar_liar.models.base_models.sequential_model import SequentialModel
from liar_liar.attacks.fgsm import fgsm
from liar_liar.utils.attack_metrics import *


class MNISTModel(SequentialModel):

#Here the dataset_name responds to names defined in tensorflow_datasets
    def __init__(self, optimizer, loss, metrics, MODEL_NAME):
        super().__init__(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics,
                         MODEL_NAME=MODEL_NAME,
                         dataset_name='mnist')


class MNISTDenseModel(MNISTModel):

    def __init__(self, optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy]):
        super().__init__(optimizer=optimizer, loss=loss, metrics=metrics, MODEL_NAME="MnistDense")
        self.sequential_layers = [
            Flatten(),
            Dense(784, activation='sigmoid'),
            Dense(800, activation='sigmoid'),
            Dense(800, activation='sigmoid'),
            Dense(10, activation='softmax')
        ]



#Create instance of your model
model = MNISTDenseModel()
#Train or load you model
model.train(epochs=15, augment_data=False)
# model.load_weights("path/to/saved/model")
batch_size = 32
nmb_batches = 10
#Get the dataset
dataset = model.get_dataset(tfds.Split.TEST, batch_size=batch_size)
#Define metrics that we want to gather
metrics_accumulator = AttackMetricsAccumulator([Accuracy(), L2_Metrics(), Robustness(), Timing()])
for data_sample in dataset.take(nmb_batches):
    image, labels = data_sample
    #Run the fgsm attack
    ret_image, logits, parameters = fgsm(model, data_sample)
    #Gather metrics
    metrics_dict = metrics_accumulator.accumulate_metrics(image, labels, ret_image, logits, batch_size)
#And print them
print(metrics_dict)
```