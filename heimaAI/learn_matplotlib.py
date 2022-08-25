import random

import matplotlib.pyplot as plt

# # 初始化xy轴,准备数据
x = range(60)
y_shanghai = [random.uniform(15, 18) for i in x]
y_beijing = [random.uniform(1, 3) for i in x]
# 创建画布
# plt.figure(figsize=(20, 8), dpi=100)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), dpi=100)

# 绘制图像
# plt.plot([1, 2, 3, 4, 5, 6, 7], [10, 101, 19, 111, 2, 11, 4])
# 图像显示
# plt.show()

#
# plt.figure(figsize=(20, 8), dpi=100)
axes[0].plot(x, y_shanghai, label='shanghai')
axes[1].plot(x, y_beijing, label='beijing', linestyle='--', color='r')
# # xy轴刻度标签
# x_tricks_label = ["11:{}".format(i) for i in x]
# y_tricks = range(40)
# # 修改坐标刻度显示
# plt.xticks(x[::5], x_tricks_label[::5])
# plt.yticks(y_tricks[::5])
# # 网格
# plt.grid(True, linestyle="--", alpha=0.5)
# # 描述信息
# plt.xlabel("time", fontsize=20)
# plt.ylabel("temp", fontsize=20)
# plt.title("city_temp")
# # 保存图像
# # plt.savefig("./1.png")
#
# # 绘制多个图像


# 显示图例
# plt.legend(loc="best")


plt.show()
