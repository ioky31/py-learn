##############################交叉验证#################################
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y=None, scoring=None, cv=None, n_jobs=1)
"""参数
---
    model：拟合数据的模型
    cv ： k-fold
    scoring: 打分参数-‘accuracy’、‘f1’、‘precision’、‘recall’ 、‘roc_auc’、'neg_log_loss'等等
"""

scores = []
alphas = []
for alpha in range(1, 100):
    knn = neighbors.KNeighborsClassifier(n_neighbors=alpha)  # 分类
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test)
    sc = np.sqrt(-cross_val_score(knn, x_train, y_train, scoring="neg_mean_squared_error", cv=10))
    print(sc.mean())
    scores.append(np.array(sc.mean()))
    alphas.append(alpha)
plt.plot(alphas, scores)
plt.show()

##############################检验曲线#################################
from sklearn.model_selection import validation_curve
train_score, test_score = validation_curve(model, X, y, param_name, param_range, cv=None, scoring=None, n_jobs=1)
"""参数
---
    model:用于fit和predict的对象
    X, y: 训练集的特征和标签
    param_name：将被改变的参数的名字
    param_range： 参数的改变范围
    cv：k-fold

返回值
---
   train_score: 训练集得分（array）
    test_score: 验证集得分（array）
"""
##############################混淆矩阵#################################
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_Actual, y_Predicted, labels=[0, 1])  # 计算混淆矩阵
print("混淆矩阵为:\nIn{}\n".format(confusion_matrix))

##############################混淆矩阵#################################
P = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])  # 计算精确率
R = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])  # 计算召回率
F1score = 2 * P * R / (P + R)  # 计算F1测度值
print("F1score: ", F1score)

##############################roc曲线#################################
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(test_label, predict)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve')
plt.legend(loc="lower right")

##############################混淆矩阵热力图#################################
import seaborn as sns
sns.set_context({"figure.figsize": (8, 8)})
sns.heatmap(data=confusion_matrix, square=True, annot=True)