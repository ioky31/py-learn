from sklearn.preprocessing import StandardScaler

X = [[0, 15],
     [1, -10]]
X_after = StandardScaler().fit(X).transform(X)

print(X_after)