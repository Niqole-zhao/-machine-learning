import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from sklearn import datasets
import numpy as np

#训练集的输入特征x_train和y_train
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

#数据集乱序
np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(3, activation="softmax",
                                                          kernel_regularizer=tf.keras.regularizers.l2())])

model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["sparse_categorical_accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

model.summary()