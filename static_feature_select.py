from stats_flow import *
import numpy as np


# 根据筛选出有显著性差异的特征
class static_feature_select:
    def __init__(self, significance_level=0.1):
        self.significance_level = significance_level

    def fit(self, X, y):
        ADdataset = X[y == 1]
        WTdataset = X[y == 0]
        length = X.shape[1]
        ps = []
        for i in range(0, length):
            _, p, _ = stats_flow(WTdataset=WTdataset.iloc[:, i], ADdataset=ADdataset.iloc[:, i],
                                 significance_level=self.significance_level)
            ps.append(p)
        ps = np.array(ps)
        self.feature_list = ps < self.significance_level

    def transform(self, X):
        return X.iloc[:, self.feature_list]
