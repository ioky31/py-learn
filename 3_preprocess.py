# ##############################数据标准化########################################
# # x = (x - mean)/std 按列即每个特征标准化
# from sklearn.preprocessing import StandardScaler
# X_after = StandardScaler().fit().transform()
#
# ##############################数据归一化########################################
# # x = (x - min)/(max - min)
# from sklearn.preprocessing import MinMaxScaler
# # copy：为拷贝属性，默认为True,表示对原数据组拷贝操作，这样变换后元数组不变，False表 示变换操作后，原数组也跟随变化，相当于c++中的引用或指针。copy = 1 的效果 同 copy = True，copy = 0 的效果同 copy =False
# MinMaxScaler(feature_range=[0, 1], copy=True)
#
# ##############################数据正则化########################################
# # 对该样本中每个元素除以该范数，这样处理的结果是使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1。
# from sklearn.preprocessing import normalize
# normalize(norm='l2')
#
# ##############################onehot编码#######################################
# # Onehot编码是分类变量作为二进制向量的表示。
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder().fit()
#
# ##############################缺失值处理#######################################
# from sklearn.impute import SimpleImputer
# imp_mean = SimpleImputer() #实例化。默认均值填补
# imp_median = SimpleImputer(strategy='median') #用中位数填补
# imp_0 = SimpleImputer(strategy='constant', fill_value=0) #用常量0填充
#
# ##############################标签编码#######################################
# from sklearn.preprocessing import LabelEncoder
# LabelEncoder = LabelEncoder() #标签专用，能够将分类转换为分类数值
#
# ##############################将分类特征转换为分类数据#######################################
# from sklearn.preprocessing import OrdinalEncoder
# OrdinalEncoder = OrdinalEncoder()#特征专用，能够将分类特征转换为分类数值
#
# ##############################二值化#######################################
# from sklearn.preprocessing import Binarizer
# Binarizer = Binarizer(threshold=35)
#
# ##############################分箱处理#######################################
# from sklearn.preprocessing import KBinsDiscretizer
# KBinsDiscretizer = KBinsDiscretizer()# 是将连续特征划分为离散特征值的方法

##############################pca#######################################
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
newX = pca.fit_transform(X)
print(pca.explained_variance_ratio_)# 显示权重
'''
n_components: int, float, None 或 string，PCA算法中所要保留的主成分个数，也即保留下来的特征个数，如果 n_components = 1，将把原始数据降到一维；如果赋值为string，如n_components='mle'，将自动选取特征个数，使得满足所要求的方差百分比；如果没有赋值，默认为None，特征个数不会改变（特征数据本身会改变）。
copy：True 或False，默认为True，即是否需要将原始训练数据复制。
whiten：True 或False，默认为False，即是否白化，使得每个特征具有相同的方差。
'''
##############################TSNE#######################################
from sklearn.manifold import TSNE
X_data_stne = TSNE(n_components=n).fit_transform(X_data)# 只能降到2-3维

##############################FactorAnalysis#######################################
from sklearn.decomposition import FactorAnalysis

##############################DictionaryLearning#######################################
from sklearn.decomposition import DictionaryLearning


