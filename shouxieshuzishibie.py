import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()    #指定输入训练集和测试集标签和特征
x_train, x_test = x_train/255.0, x_test/255.0      #输入特征进行归一化处理，使原来在0-255之间的灰度值，变为0-1之间的数值，
                                                    # 使输入特征数值变小，更适合神经网络吸收

#用sequential搭建网络
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),     #先将输入特征拉直为以为数组
    tf.keras.layers.Dense(128, activation="relu"),     #定义第一层网络128个神经元，激活函数relu
    tf.keras.layers.Dense(10, activation="softmax"),        #定义第二层网络10个神经元，激活函数softmax,使输出符合概率分布
])

model.compile(optimizer="adam",    #compile配置训练方法，优化器选择Adam
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), #loss：SparseCategoricalCrossentropy
              #由于上面已经让输出符合概率分布，因此不是直接输出，有from_logits=False，如果输出不满足概率分布则写True
              metrics=["sparse_categorical_accuracy"])  #数据集标签为数值，神经网络输出y是概率分布，所以选择：sparse_categorical_accuracy

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
#validation_freq=1表示每迭代一次训练集执行一次测试集评测
model.summary()   #打印出网络结构和参数统计