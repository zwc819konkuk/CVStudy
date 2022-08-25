import matplotlib.pyplot as plt

x = [12, 1, 4, 3]
y = [1, 4, 3, 1]

plt.figure(figsize=(20, 8), dpi=100)
# plt.scatter(x,y)
plt.bar(x, y, color=['r', 'b', 'c', 'm'],width=0.7)
plt.show()
