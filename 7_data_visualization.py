#数据可视化
import matplotlib.pyplot as plt
plt.imshow(digits.images[-1], cmap=plt.cm.gray)
# plt.show()
# digits.images.shape: (1797, 8, 8)
# 散点图
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
#plt
import matplotlib.pyplot as plt
plt.plot(alphas_log, coefs_T[row, :], ls="-", lw=2, label="plot figure")
plt.legend()
plt.show()
