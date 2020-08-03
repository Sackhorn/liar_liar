import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from liar_liar.attacks.deepfool import deepfool
from liar_liar.utils.attack_metrics import *


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
                   Dense(10, activation='softmax'),Dense(10, activation='softmax')])

model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy])
model.fit(train.repeat(),
          epochs=5,
          steps_per_epoch=train_steps,
          validation_data=test,
          validation_steps=test_steps)

#Define metrics that we want to gather
metrics_accumulator = AttackMetricsAccumulator([Accuracy(), L2_Metrics(), Robustness(), Timing()])
for data_sample in test.take(10):
    image, labels = data_sample
    ret_image, logits, parameters = deepfool(model, data_sample, iter_max=100) #Run the attack
    metrics_dict = metrics_accumulator.accumulate_metrics(image, labels, ret_image, logits, BATCH_SIZE)
print(metrics_dict)
