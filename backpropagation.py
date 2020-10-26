import tensorflow as tf

w = tf.Variable(tf.constant(5,dtype=tf.float32))   #设定参数w的随机初始值为5
                                                  # float32区别于float64,数位的区别，其中64占的内存大
lr = 0.2
epoch = 40     #迭代次数40次

for epoch in range(epoch):   # for epoch定义顶层循环，表示对数据集循环epoch次，
    """用with结构让损失函数loss对参数w求梯度"""
    with tf.GradientTape() as tape:  # with结构到grads框起了梯度的计算过程
        loss = tf.square(w + 1)   #定义损失函数loss
    grads = tape.gradient(loss, w)   # .gradient函数告知谁对谁求导

    w.assign_sub(lr * grads)     # .assign_sub对变量做自减 即：w -= lr*grads
    print("After %s epoch, w is %f" % (epoch, w.numpy(), loss))

# lr初始值：0.2  学习率范围0.001到0.999
# 最终目的：找到 loss 最小 即w = -1的最优参数w