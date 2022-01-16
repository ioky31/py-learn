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
