"""创建一个张量"""
import tensorflow as tf
a = tf.constant([1,5], dtype= tf.int64)
print(a)
print(a.dtype)
print(a.shape)

"""将numpy的数据类型转换为Tensor数据类型"""
import numpy as np
b= np.arange(0,5)
c= tf.convert_to_tensor(a, dtype= tf.int64)
print(b)
print(c)

"""创建指定值的张量"""
d= tf.zeros([2,3])
e= tf.ones(4)
f= tf.fill([2,2],9)
print(d,"\n",e,"\n",f)

"""生成正太分布随机数"""
g= tf.random.normal([2,2], mean=2, stddev=4)
h= tf.random.truncated_normal([2,3], mean=2, stddev=4)
print(g,"\n",h)

"""常用函数"""
x1=tf.constant([1,2,3], detype= tf.float(64))
print(x1)
x2=tf.cast(x1, tf.int32)
print(x2)
print(tf.reduce_min(x2), tf.educe_max(x2))