#导入模块
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#k近邻函数
iris = datasets.load_iris()
#导入数据和标签
iris_X = iris.data
iris_y = iris.target
print(iris_y.shape)
#划分为训练集和测试集数据(test_size比例， random_state随机种子， stratify保持分布)
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3, random_state=0, stratify=iris_y)
#print(y_train)
#设置knn分类器
knn = KNeighborsClassifier()
#进行训练
knn.fit(X_train,y_train)
#使用训练好的knn进行数据预测
# print(knn.predict(X_test))
# print(y_test)