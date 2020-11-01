from sklearn import datasets
from pandas import DataFrame
import pandas as pd

x_data = datasets.load_iris().data     #.data返回iris数据集所有输入特征
y_data = datasets.load_iris().target       #.target返回iris数据集所有标签
"""上两句返回所有输入特征和标签"""
print("x_data from datasets:\n", x_data)
print("y_data from datasets:\n", y_data)

"""为了增加可读性，用DataFrame()把数据变成表格形式，每一列增加中文标签"""
x_data = DataFrame(x_data, columns=["花萼长度","花萼宽度","花瓣长度","花瓣宽度"])
pd.set_option("display.unicode.east_asian_width",True)    #设置列名对齐
print("x_data add index:\n", x_data)

x_data["类别"] = y_data   #新加一列，列标签为“类别，数据为y_data
print("x_data add a column:\n", x_data)

#类型维度不确定时，建议用print函数出来确认效果