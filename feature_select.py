# 根据模型选择特征
from sklearn.feature_selection import SelectFromModel

# 模型
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1).fit(X_data, y_data)
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_data, y_data)
clf = ExtraTreesClassifier().fit(X_data, y_data)

model = SelectFromModel(forest, prefit=True)
feature_idx = model.get_support()
feature_name = X_data.columns[feature_idx]
X_data = model.transform(X_data)
X_data = pd.DataFrame(X_data)
X_data.columns = feature_name

# RFE递归消除
from sklearn.feature_selection import RFE

rfe = RFE(lr, n_features_to_select=1)
rfe.fit(X_data, y_data)

# 根据交叉验证rfe筛选特征
def my_rfecv(X_train, y_train, need_print=False, need_draw=False):
    svc = SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy')
    rfecv.fit(X_train, y_train)

    if need_print:
        print("Optimal number of features : %d" % rfecv.n_features_)
        print("Ranking of features : %s" % rfecv.ranking_)
        print(rfecv.grid_scores_)

    if need_draw:
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
    return rfecv.transform(X_train)
