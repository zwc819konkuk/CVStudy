# 绘图
# 数值计算
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
# 机器学习
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from tensorflow.keras.layers import Dense
# 深度学习
from tensorflow.keras.models import Sequential

# 1. 数据处理
# 读取数据
iris = sns.load_dataset("iris")
# 展示数据的前五行
# print(iris.head())
# sns.pairplot(iris,hue="species")
# 获取数据的特征值和目标值
X = iris.values[:, :4]
y = iris.values[:, 4]
# 数据集划分
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=0)
print(train_X.shape)
print(test_X.shape)

# 2. sklearn实现
# 实例化估计器【分类器】 逻辑回归
lr = LogisticRegressionCV()
# 训练
lr.fit(train_X, train_y)
# 模型评估 计算准确率并打印
print(lr.score(test_X, test_y))


# 3. tf.keras实现
# 数据处理
# 目标值实现热编码
def one_hot_encode(arr):
    # 获取目标值中的所有类别并进行热编码
    uniques, ids = np.unique(arr, return_inverse=True)
    return utils.to_categorical(ids, len(uniques))


# 对目标值进行编码
train_y_ohe = one_hot_encode(train_y)
test_y_ohe = one_hot_encode(test_y)

# 模型构建
model = Sequential([
    # 隐藏层01
    Dense(10, activation="relu", input_shape=(4,)),
    # 隐藏层02
    Dense(10, activation="relu"),
    # 输出层
    Dense(3, activation="softmax")
])
print(model.summary())
utils.plot_model(model, show_shapes=True)


# 模型训练和预测
# 设置模型的相关参数：优化器，损失函数和评价指标
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# 类型转换
train_X = np.array(train_X, dtype=np.float32)
test_X = np.array(test_X, dtype=np.float32)
# 模型训练
model.fit(train_X, train_y_ohe, epochs=10, batch_size=1, verbose=1)
# 模型评估
loss, accuracy = model.evaluate(test_X, test_y_ohe, verbose=1)
print(loss, accuracy)
