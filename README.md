Liar Liar - Adversarial Attacks Testing Suite
=============================================
Liar Liar is a library based on Tensorflow 2 and Keras implementing multiple known adversarial attacks.\
It was created as a part of my Bachelor's thesis.\
It's goal is to provide a simple API for testing models resistance and for creating samples for adversarial training.

Requirements
______________
- Python 3.x
- You can find information on setting up Tensorflow with GPU support [here](https://www.tensorflow.org/install/gpu).

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
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from liar_liar.attacks.fgsm import FGSMUntargeted
from liar_liar.utils.attack_metrics import *
from liar_liar.utils.dataset_creator import *


def preprocess_data(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.one_hot(y, 10)
    return x, y

#Get dataset
BATCH_SIZE = 32
(train, test), info = tfds.load(name="mnist",
                                split=[tfds.Split.TRAIN, tfds.Split.TEST],
                                with_info=True,
                                as_supervised=True)
train = train.map(preprocess_data).batch(BATCH_SIZE)
test = test.map(preprocess_data).batch(BATCH_SIZE)
test_steps = info.splits['test'].num_examples // BATCH_SIZE
train_steps = info.splits['train'].num_examples // BATCH_SIZE

#Define and train your model
model = Sequential([Flatten(),
                   Dense(784, activation='sigmoid'),
                   Dense(800, activation='sigmoid'),
                   Dense(800, activation='sigmoid'),
                   Dense(10, activation='softmax')])

model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy])
model.fit(train.repeat(),
          epochs=5,
          steps_per_epoch=train_steps,
          validation_data=test,
          validation_steps=test_steps)

#Define metrics that we want to gather
metrics_accumulator = AttackMetricsAccumulator([Accuracy(), L2_Metrics(), Robustness(), Timing()])
for data_sample in test.take(100):
    image, labels = data_sample
    fgsm = FGSMUntargeted(iter_max=100, eps=0.001)
    adv_image, adv_logits, parameters = fgsm(model, data_sample) #Run the attack
    metrics_dict = metrics_accumulator.accumulate_metrics(image, labels, adv_image, adv_logits, BATCH_SIZE)
print(metrics_dict)

#Create adversary datasets for both test and train datasets
test_adv_dataset = create_adv_dataset(test, FGSMUntargeted, model, {ITER_MAX: 100, EPS: 0.001})
train_adv_dataset = create_adv_dataset(train, FGSMUntargeted, model, {ITER_MAX: 100, EPS: 0.001})
save_adv_dataset(test_adv_dataset, "test_adv_mnist")
save_adv_dataset(train_adv_dataset, "train_adv_mnist")

```
You should see something like this
```
{'accuracy_result': 0.9993750000139698, 'L2_average': 1.2045974825398298, 'L2_median': 1.2455607652664185, 'robustness_key': 0.1307359296904724, 'time_per_batch': 0.1600356101989746, 'average_time_per_sample': 0.007932707667350769}
```
and an image
![comparing adversarial example and an original](https://github.com/Sackhorn/liar_liar/blob/master/readme_figure.png "Comparison"")
if you want to generate a dataset of adversarial examples you can just do this
```
test_adv_dataset = create_adv_dataset(test, fgsm_untargeted_wrapper, model, {ITER_MAX: 100, EPS: 0.001})
train_adv_dataset = create_adv_dataset(train, fgsm_untargeted_wrapper, model, {ITER_MAX: 100, EPS: 0.001})
save_adv_dataset(test_adv_dataset, "test_adv_mnist")
save_adv_dataset(train_adv_dataset, "train_adv_mnist")
```
this is a regular TensorFlow dataset so you can perform all the operations normally available
If you want to load the saved dataset just call ```load_adv_dataset()```
```python
test_adv_dataset = load_adv_dataset("test_adv_mnist")
train_adv_dataset = load_adv_dataset("train_adv_mnist")
```
If you want to run tests then just run this script
```liar_liar/liar_liar/tests/run_sanity_checks.py```
