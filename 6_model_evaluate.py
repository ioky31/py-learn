##############################手动k折交叉验证#################################
from sklearn.model_selection import KFold
for train_indices, test_indices in kf.split(data_X):
    MLPClassifier_.fit(data_X.iloc[train_indices], data_y[train_indices])
    print(MLPClassifier_.score(data_X.iloc[test_indices], data_y[test_indices]))#     dataFrame数据类型要加。iloc才能索引
    
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

##############################误差计算#################################
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, pred)
'''
~~'accuracy_score',~~ 
~~'adjusted_mutual_info_score',~~ 
'adjusted_rand_score',
'auc',
'average_precision_score',
'balanced_accuracy_score',
'calinski_harabasz_score',
'check_scoring',
'classification_report',
'cluster',
'cohen_kappa_score',
'completeness_score',
'ConfusionMatrixDisplay',
'confusion_matrix',
'consensus_score',
'coverage_error',
'dcg_score',
'davies_bouldin_score',
'DetCurveDisplay',
'det_curve',
'euclidean_distances',
'explained_variance_score',
'f1_score',
'fbeta_score',
'fowlkes_mallows_score',
'get_scorer',
'hamming_loss',
'hinge_loss',
'homogeneity_completeness_v_measure',
'homogeneity_score',
'jaccard_score',
'label_ranking_average_precision_score',
'label_ranking_loss',
'log_loss',
'make_scorer',
'nan_euclidean_distances',
'matthews_corrcoef',
'max_error',
~~'mean_absolute_error',~~ 
'mean_squared_error',
'mean_squared_log_error',
'mean_poisson_deviance',
'mean_gamma_deviance',
'mean_tweedie_deviance',
'median_absolute_error',
'mean_absolute_percentage_error',
'multilabel_confusion_matrix',
'mutual_info_score',
'ndcg_score',
'normalized_mutual_info_score',
'pair_confusion_matrix',
'pairwise_distances',
'pairwise_distances_argmin',
'pairwise_distances_argmin_min',
'pairwise_distances_chunked',
'pairwise_kernels',
'plot_confusion_matrix',
'plot_det_curve',
'plot_precision_recall_curve',
'plot_roc_curve',
'PrecisionRecallDisplay',
'precision_recall_curve',
'precision_recall_fscore_support',
~~'precision_score',~~ 
'r2_score',
'rand_score',
~~'recall_score',~~ 
'RocCurveDisplay',
'roc_auc_score',
'roc_curve',
'SCORERS',
'silhouette_samples',
'silhouette_score',
'top_k_accuracy_score',
'v_measure_score',
'zero_one_loss',
'brier_score_loss',
'''
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
confusion_matrix_ = confusion_matrix(y_Actual, y_Predicted, labels=[0, 1])  # 计算混淆矩阵
print("混淆矩阵为:\nIn{}\n".format(confusion_matrix_))

##############################混淆矩阵#################################
P = confusion_matrix_[1, 1] / (confusion_matrix_[1, 1] + confusion_matrix_[0, 1])  # 计算精确率
R = confusion_matrix_[1, 1] / (confusion_matrix_[1, 1] + confusion_matrix_[1, 0])  # 计算召回率
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

##############################PR曲线#################################
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure("P-R Curve")
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall, precision)
plt.show()

##############################混淆矩阵热力图#################################
import seaborn as sns
sns.set_context({"figure.figsize": (8, 8)})
sns.heatmap(data=confusion_matrix_, square=True, annot=True)
