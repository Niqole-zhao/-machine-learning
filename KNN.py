import matplotlib.pyplot as plt  #matplotlib用于数据的展示
import numpy as np
#-------------视觉图分析-------------
a=np.array([[3,104],
          [2,100],
          [1,81],
          [101,10],
          [99,5],
          [98,2]])
for i in range(3):
    plt.plot([18,a[i,0]],[90,a[i,1]],color="r")
    plt.scatter([18,a[i,0]],[90,a[i,1]],color="r")
for i in range (3,6):
    plt.plot([18, a[i, 0]], [90, a[i, 1]], color="b")
    plt.scatter([18, a[i, 0]], [90, a[i, 1]], color="b")
plt.scatter(18,90,color="y")
plt.show()
# -------------KNN核心代码-------------
# inX用于分类的输入向量   c
#    dataSet输入的训练样本集  a
#    labels标签向量    b
#    k用于选择的最邻近的数目   4
def KNN(inX,dataSet,labels,k):
   dataSetSize =dataSet.shape[0]   #用于获取dataSet的行列数，结果为6，
   print("dataSetSize:",dataSetSize)
   diffMat =np.tile(inX,(dataSetSize,1))- dataSet  #第一步：计算已知类别数据集中的，此处将普通的列表转换为array再相减
   print(np.tile(inX,(dataSetSize,1)))
   print("diffMat:\n",diffMat)
   sqDiffMat =diffMat **2
   print("sqDiffMat:\n",sqDiffMat)
   aqDistances =sqDiffMat.sum(axis=1)
   print("aqDistances:\n",aqDistances)
   distances=aqDistances**0.5
   print("distances:\n",distances)
   sortedDistlandicies =distances.argsort()   #第二步，从小到大排序
   print("sortdDistIndicies",sortedDistlandicies)
   classCount={}  #定义一个空字典
   for i in range (k):  #第三步选取与当前点距离最小的K个点
       votellabel =labels[sortedDistlandicies[i]]
       print("voteIlabel:",votellabel)
       classCount[votellabel]=classCount.get(votellabel,0)+1  #第四步确定前k个点所在类别的出现频率
       print("sortedClassCount:",classCount)
   print("items:",classCount.items())
   sortedClassCount = list(classCount.items())    #第五步返回前k个点出现频率最高的类别作为当前点的预测分类
   print("sortedClassCount:",sortedClassCount)
   sortedClassCount.sort(key=lambda x:x[1],reverse=True)   #sort函数用于进行排序
   print(sortedClassCount[0][0])
   return sortedClassCount[0][0]   #返回频率最高的第一个元素

a=np.array([[3,104],
            [2, 100],
            [1, 81],
            [101, 10],
            [99, 5],
            [98, 2]])
b=["爱情片","爱情片","爱情片","动作片","动作片","动作片"]
c=[18,90]
result_lab=KNN(c,a,b,4)

#--------------------部分函数解析--------------------------
#shape方法解析：shape是属于numpy库中的array对象的方法，用于读取数据对象的行数和列数，以元组的形式返回
# print(a.shape)  #打印结果为（6,2），a为6行，2列
# print(a.shape[1])  #获取列数
# print(a.shape[0])  #获取行数
#tile方法：将数组重复n次，构建一个新的数组
# e=np.array([[1,2,3]])
# f=np.array([[1,2,3],[2,3,4]])
# print("e的结果为：\n",np.tile(e,2))
# print("f的结果为：\n",np.tile(f,2))
# #假如我们输入元组（1,2）
# print("f的结果为：\n",np.tile(f,(2,2)))
#sum方法解析：sum函数是属于numpy库中的array对象方法，用于求和
# g=np.array([[1,2,3],[9,8,7]])
# print(g.sum())  #axis=None,将数组/矩阵中的元素全部加起来，得到一个和
# print(g.sum(axis=1))   #axis=1，将每一行的元素相加，将矩阵压缩为一列
# print(g.sum(axis=0))    #axis=0，将每一列的元素相加，将矩阵压缩为一行

#argsort方法，属于numpy库中array对象的方法，功能：数返回的是数组从小到大的索引值
# h=np.array([3,1,2,0,5])
# print(np.argsort(h))

#sort方法：对列表进行排序
# x=[4,6,2,1,-7,9]
# x.sort()   #对列表进行从小到大排序
# x.sort(reverse = True)  #对列表进行从大到小的排序
# x.sort(key = abs)  #abs对元素取绝对值。排序方法绝对值从小到大排序
# x.sort(key = abs,reverse = True)   #绝对值从大到小排序
# print(x)

# d={"张三":89,"李四":92,"王五":87}
# d_list =list(d.items())
# print(d_list)
# d_list.sort(key = lambda x:x[1],reverse=True)
# print(d_list)