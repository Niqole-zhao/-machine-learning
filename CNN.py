#--------------------网络框架--------------------------
import numpy
import scipy.special
import matplotlib.pyplot
class  neuralNetwork:
    def __inint__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #设置输入层、隐藏层、输出层的节点数
        self.indoes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #设置学习率
        self.lr=learningrate
        #定义激励函数，lambda直接定义函数，调用scipy
        self.activation_function=lambda x:scipy.special.expit(x)

        pass

    def train(self,inputs_list,targets_list):
        #把输入的列表编程2维数列
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T
        #正向传播
        #计算隐藏层输入的信号
        hidden_inputs=numpy.dot(self.wih,inputs)
        #计算隐藏层输出的信号
        hidden_outputs=self.activation_function(hidden_inputs)

        #计算最终层输入的信号
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #计算最终层输出的信号
        final_outputs=self.activation_function(final_inputs)
        #反向传播
        #计算最终层的误差
        output_errors=targets-final_outputs
        #计算隐藏层的误差
        hidden_errors=numpy.dot(self.who.T,output_errors)
        #更新隐藏层和输出层的节点权重
        self.whi +=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),
                                     numpy.transpose(hidden_outputs))
        #更新输入层和隐藏层的节点的权重
        self.wih +=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),
                                     numpy.transpose(inputs))

        pass
    def query(self,inputs_list):
        #把输入的列表变成2维数列
        inputs=numpy.array(inputs_list,ndmin=2).T
        #计算隐藏层的输入和输出信号
        hidden_inputs=numpy.dot(self.win,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        #计算最终层的输入输出信号
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        return final_outputs
# --------------------设置网络参数--------------------------
#设置神经网络参数，节点数，学习率
input_nodes=784
hidden_nodes=200
output_nodes=10
learning_rate=0.2
n=neuralNetwork()
print(input_nodes,hidden_nodes,output_nodes,learning_rate)
#导入训练集
training_data_file=open("E:\PycharmFiles\witing1\mnist_train.csv","r")
training_data_list=training_data_file.readlines()
training_data_file.close()
len(training_data_list)
#训练神经网络，计算耗时比较长
epochs=5
for e in range(epochs):
    for record in training_data_list:
        all_values=record.split(",")
        inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets=numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
        pass
    pass
#导入测试数据
test_data_file=open("mnist_test.csv","r")
test_data_list=test_data_file.readlines()
test_data_file.close()
#测试神经网络，用test测试集数据测试神经网络正确率
#定义scorecard别忘了
scorecard=[]
for record in test_data_list:
    all_vlues=record.split(",")
    correct_label=int(all_values[0])
    print(correct_label,"correct label")
    inputs =(numpy.asfarray(all_values[1:])/255.0*0.99+0.01)
    outputs=n.query(inputs)
    #取神经网络输出层的最大值为结果
    label=numpy.argmax(outputs)
    print(label,"network's answer")
    if(label ==correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass
#评价神经网络的表现，结果代表神经网络识别数字判断正确的百分比
scorecard_array=numpy.asarray(scorecard)
print("performance=",scorecard_array.sum()/scorecard_array.size)
