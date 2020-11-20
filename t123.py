import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

train_images.shape# 查看数据集

input = keras.Input(shape=(28, 28))# 建立一个输入模型（形状28*28）

# 调用Flatten层，可以把keras.layers.Flatten()看作一个函数参数input
x = keras.layers.Flatten()(input)
# 调用dense层输出32个隐藏单元 激活函数relu 参数x
x = keras.layers.Dense(32,activation="relu")(x)
# 添加一个印制拟合Dropout层
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(64,activation="relu")(x)

output = keras.layers.Dense(10,activation="softmax")(x)  # 建立一个输出模型
# 建立模型
model = keras.Model(inputs=input,outputs=output)

# 模型的形状
model.summary()
#    [(None, 28, 28)]   None 表示任意值
# 编译模型
model.compile(optimizer="adam",
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"]
             )
# 训练模型
history = model.fit(train_images,
                   train_labels,
                   epochs=30,
                   validation_data=(test_images,test_labels))
