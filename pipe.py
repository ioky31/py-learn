from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 创建一个pipeline对象
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=0)
)
# 加载鸢尾花数据集并将其切分成训练集和测试集
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 训练整个pipeline
pipe.fit(X_train, y_train)
