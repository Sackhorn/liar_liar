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
