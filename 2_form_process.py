
#将图像转换为特征向量(n, rows, cols)=>(n, vector)
data = data.reshape(data.shape[0], -1)

#划分为训练集和测试集数据(test_size比例， random_state随机种子， stratify保持分布)
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3, random_state=0, stratify=iris_y)
