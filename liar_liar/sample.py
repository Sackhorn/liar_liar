import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from liar_liar.attacks.deepfool import deepfool
from liar_liar.utils.attack_metrics import *
from liar_liar.utils.generate_side_by_side import show_plot_comparison


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
def nmb_classes():
    return 10
model.get_number_of_classes = nmb_classes
#Define metrics that we want to gather
metrics_accumulator = AttackMetricsAccumulator([Accuracy(), L2_Metrics(), Robustness(), Timing()])
for data_sample in test.take(100):
    image, labels = data_sample
    adv_image, adv_logits, parameters = deepfool(model, data_sample, iter_max=100) #Run the attack
    metrics_dict = metrics_accumulator.accumulate_metrics(image, labels, adv_image, adv_logits, BATCH_SIZE)
print(metrics_dict)
show_plot_comparison(adv_image[0],
                     adv_logits[0],
                     image[0],
                     model(tf.expand_dims(image[0], 0)),
                     list(range(10)),
                     true_class=labels[0])
