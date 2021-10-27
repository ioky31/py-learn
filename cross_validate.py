from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0)
lr = LinearRegression()

result = cross_validate(lr, X, y)  # 默认为5折交叉验证

print(result['test_score'])  # 此处R2得分很高的原因为数据集很简单
