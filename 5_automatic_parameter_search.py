###################################随机搜索############################################
'''
少量离散参数用网络搜索
多连续参数用随机搜索
'''
# import numpy as np
# from scipy.stats import randint as sp_randint
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.datasets import load_digits
# from sklearn.ensemble import RandomForestClassifier
#
# # 载入数据
# digits = load_digits()
# X, y = digits.data, digits.target
#
# # 建立一个分类器或者回归器
# clf = RandomForestClassifier(n_estimators=20)
#
# # 给定参数搜索范围：list or distribution
# param_dist = {"max_depth": [3, None],                     #给定list
#               "max_features": sp_randint(1, 11),          #给定distribution
#               "min_samples_split": sp_randint(2, 11),     #给定distribution
#               "bootstrap": [True, False],                 #给定list
#               "criterion": ["gini", "entropy"]}           #给定list
#
# # 用RandomSearch+CV选取超参数
# n_iter_search = 20
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search, cv=5)
# random_search.fit(X, y)

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

# k近邻函数
iris = datasets.load_iris()
# 导入数据和标签
iris_X = iris.data
iris_y = iris.target

# 样本矩阵（或设计矩阵）X。X的大小通常为(n_samples, n_features)，这意味着样本表示为行，特征表示为列。
# 目标值y是用于回归任务的真实数字，或者是用于分类的整数（或任何其他离散值）。对于无监督学习，y无需指定。y通常是1d数组，其中i对应于目标X的 第i个样本（行）。
# 划分为训练集和测试集数据(test_size比例， random_state随机种子， stratify保持分布)
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3, random_state=0, stratify=iris_y)
# 设置knn分类器
knn = KNeighborsClassifier()
# 给定参数搜索范围：list or distribution
param_dist = {"n_neighbors": range(1, None),  # 给定list
              "weights": ['uniform', 'distance'],
              "metric": ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}  # 给定list
# 用RandomSearch+CV选取超参数
n_iter_search = 100
random_search = RandomizedSearchCV(knn, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)# cv是交叉验证的次数
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
print(knn.score(X_test, y_test))
best_estimator = random_search.best_estimator_
print(best_estimator)
#输出最优训练器的精度
print(random_search.best_score_)
###################################网络搜索############################################
from sklearn.model_selection import GridSearchCV

param_grid = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'solver': ['lbfgs', 'sgd', 'adam'],
              'hidden_layer_sizes': [(50,), (100,), (150,), (200,)]}

MLPClassifier_ = MLPClassifier(activation='relu', solver='adam', alpha=0.0001, max_iter=100000)
grid_search = GridSearchCV(MLPClassifier_, param_grid, cv=10,
                           scoring='accuracy')
grid_search.fit(X_train, y_train)
best_estimator = grid_search.best_estimator_
print(best_estimator)
#输出最优训练器的精度
print(grid_search.best_score_)
