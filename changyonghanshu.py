"""常用函数2"""
import tensorflow as tf
# x=tf.constant([[1,2,3],
#                [2,2,3]])
# print(x)
# print(tf.reduce_mean(x))
# print(tf.reduce_sum(x,axis=1))

"""标记可训练向量，常用于神经网络"""
# w= tf.Variable(tf.random.normal([2,2],mean=0,stddev=1))

"""标签和特征配对函数"""
# features= tf.constant([12,23,10,17])
# labels= tf.constant([0,1,1,0])
# dataset= tf.data.Dataset.from_tensor_slices((features,labels))
# print(dataset)
# for element in dataset:
#     print(element)

"""对函数求导"""
# with tf.GradientTape() as tape:
#     w = tf.Variable(tf.constant(3.0))
#     loss = tf.pow(w,2)
# grad = tape.gradient(loss,w)
# print(grad)

"""枚举，结果组合为：索引，元素"""
# seq= ["one", "two", "three"]
# for i, element in enumerate(seq):
#     print(i, element)

"""独热码,分类"""
# classes = 3
# labels = tf.constant([1, 0, 2])
# output = tf.one_hot(labels, depth=classes)
# print(output)

"""使输出符合概率分布值，tf.nn.softmax()"""
y = tf.constant([1.01, 2.02, -0.66])
y_pro = tf.nn.softmax(y)
print("After softmax, y_pro is:", y_pro)