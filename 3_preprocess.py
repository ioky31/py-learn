##############################数据标准化########################################
# x = (x - mean)/std 按列即每个特征
from sklearn.preprocessing import StandardScaler
X_after = StandardScaler().fit().transform()

##############################数据归一化########################################
# x = (x - min)/(max - min)
from sklearn.preprocessing import MinMaxScaler
# copy：为拷贝属性，默认为True,表示对原数据组拷贝操作，这样变换后元数组不变，False表 示变换操作后，原数组也跟随变化，相当于c++中的引用或指针。copy = 1 的效果 同 copy = True，copy = 0 的效果同 copy =False
MinMaxScaler(feature_range=[0, 1], copy=True)

##############################数据正则化########################################
# 对该样本中每个元素除以该范数，这样处理的结果是使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1。
from sklearn.preprocessing import normalize
normalize(norm='l2')

##############################onehot编码#######################################
# Onehot编码是分类变量作为二进制向量的表示。
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder().fit()

##############################缺失值处理#######################################
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer() #实例化。默认均值填补
imp_median = SimpleImputer(strategy='median') #用中位数填补
imp_0 = SimpleImputer(strategy='constant', fill_value=0) #用常量0填充

##############################标签编码#######################################
from sklearn.preprocessing import LabelEncoder
LabelEncoder = LabelEncoder() #标签专用，能够将分类转换为分类数值

##############################将分类特征转换为分类数据#######################################
from sklearn.preprocessing import OrdinalEncoder
OrdinalEncoder = OrdinalEncoder()#特征专用，能够将分类特征转换为分类数值

##############################二值化#######################################
from sklearn.preprocessing import Binarizer
Binarizer = Binarizer(threshold=35)

##############################分箱处理#######################################
from sklearn.preprocessing import KBinsDiscretizer
KBinsDiscretizer = KBinsDiscretizer()# 是将连续特征划分为离散特征值的方法