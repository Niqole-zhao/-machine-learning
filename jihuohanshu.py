#解决显卡内存不足的问题
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np

SEED = 2345    #定义相同的随机种子,保证每次生成的数据集一样

rdm = np.random.RandomState(seed=SEED)   #生成[0,1)之间的随机数。区间左闭右开
x = rdm.rand(32, 2)    #生成32行2列的输入x
y_ = [[x1 + x2 + (rdm.rand() /10.0 - 0.05)] for (x1, x2) in x]   #.rand()生成[0,1)之间的随机数，噪声在[0,1)之间
x = tf.cast(x, dtype=tf.float32)   #x转变数据类型

w1 = tf.Variable(tf.random.normal([2,1], stddev=1, seed=1))    #随机初始化w1，初始化为2行1列

epoach = 15000   #迭代次数
lr = 0.002

#for循环中用with结构求前向传播y和损失函数
for epoch in range(epoach):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)    #求前向传播结果y
        loss_mse = tf.reduce_mean(tf.square(y_ - y))    #求均方误差损失函数

    grads = tape.gradient(loss_mse, w1)    #损失函数对待训练参数求偏导
    w1.assign_sub(lr * grads)   #更新参数w1

#每迭代500轮，更新参数w1
    if epoch % 500 == 0:
        print("After %d training steps, w1 is " % (epoch))
        print(w1.numpy(), "\n")
print("Final w1 is: ", w1.numpy())