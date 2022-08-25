import numpy as np
from sklearn.datasets import load_boston

# 数据加载
data = load_boston()
X_ = data['data']
y = data['target']
# print(X_)
# print(y)
y = y.reshape(y.shape[0], 1)

# 数据规范化
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10
w1 = np.random.randn(n_features, n_hidden)
b1 = np.zeros(n_hidden)
w2 = np.random.randn(n_hidden, 1)
b2 = np.zeros(1)


# relu
def Relu(x):
    result = np.where(x < 0, 0, x)
    return result


def MSE_loss(y, y_hat):
    return np.mean(np.square(y_hat - y))


learning_rate = 1e-6


def Linear(X, w1, b1):
    y = X.dot(w1) + b1
    return y


for t in range(5000):
    # 前向传播
    l1 = Linear(X_, w1, b1)
    s1 = Relu(l1)
    y_pred = Linear(s1, w2, b2)

    # 计算损失函数
    loss = MSE_loss(y, y_pred)
    print(t, loss)

    # 反向传播
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = s1.T.dot(grad_y_pred)
    grad_temp_relu = grad_y_pred.dot(w2.T)
    grad_temp_relu[l1 < 0] = 0
    grad_w1 = X_.T.dot(grad_temp_relu)

    # 权重更新
    w1 = w1 - learning_rate * grad_w1
    w2 = w2 - learning_rate * grad_w2

print('w1={}\n w2={}'.format(w1, w2))
