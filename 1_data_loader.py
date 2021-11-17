###################################读取excel文件###################################
import pandas as pd
filename = "C:\\Users\\86153\\Desktop\\python与机器学习\\Class4\\class4\\stats_roc.xls"
dataset = pd.read_excel(filename)
GT = dataset['GT']
predict = dataset['predict']
# Pandas.series 会默然用0到n-1来作为series的index, 但也可以自己指定index( 可以把index理解为dict里面的key )

# series转换为numpy
import numpy as np
GT = np.array(GT)

###################################读取csv文件###################################

data = pd.read_csv(r'data.csv', index_col = 0) #数据第一列为索引，如果不加index_col=0，则会在原数据前再加一列索引

# 将 Salary 列作为 y，其他列作为 X
cols_name = data.columns.values.tolist()  # 行名称
cols_name.remove('Salary')
x_data = data[cols_name]
y_data = data['Salary']
###################################dropna()方法-丢弃含空值的行、列###################################
'''
函数形式：dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
=======
data = pd.read_csv(r'data.csv', index_col=0) #数据第一列为索引，如果不加index_col=0，则会在原数据前再加一列索引
>>>>>>> ad9754bc6520c19cad8f73239ca1cb321df8bdbf

参数：

axis：轴。0或'index'，表示按行删除；1或'columns'，表示按列删除。

how：筛选方式。‘any’，表示该行/列只要有一个以上的空值，就删除该行/列；‘all’，表示该行/列全部都为空值，就删除该行/列。

thresh：非空元素最低数量。int型，默认为None。如果该行/列中，非空元素数量小于这个值，就删除该行/列。

subset：子集。列表，元素为行或者列的索引。如果axis=0或者‘index’，subset中元素为列的索引；如果axis=1或者‘column’，subset中元素为行的索引。由subset限制的子区域，是判断是否删除该行/列的条件判断区域。

inplace：是否原地替换。布尔值，默认为False。如果为True，则在原DataFrame上进行操作，返回值为None。
'''

d.dropna(axis=0, how='any')
####################################sklearn数据库读取####################################
from sklearn import datasets
iris = datasets.load_iris()
# 导入数据和标签
iris_X = iris.data
iris_y = iris.target

####################################pickle包读取########################################
import os
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def CreatData():
    # 创建训练样本
    # 依次加载batch_data_i,并合并到x,y
    x = []
    y = []
    for i in range(1, 6):
        batch_path = 'cifar-10-batches-py\data_batch_%d' % (i)
        batch_dict = unpickle(batch_path)
        train_batch = batch_dict[b'data'].astype('float')
        train_labels = np.array(batch_dict[b'labels'])
        x.append(train_batch)
        y.append(train_labels)
    # 将5个训练样本batch合并为50000x3072，标签合并为50000x1
    # np.concatenate默认axis=0，为纵向连接
    traindata = np.concatenate(x)
    trainlabels = np.concatenate(y)
    testpath = os.path.join('cifar-10-batches-py', 'test_batch')
    test_dict = unpickle(testpath)
    testdata = test_dict[b'data'].astype('float')
    testlabels = np.array(test_dict[b'labels'])

    return traindata, trainlabels, testdata, testlabels
