import numpy as np

score = np.random.randint(40, 100, (10, 5))

# print(score)

test_score = score[6:, 0:5]
print(test_score)
test_score[test_score>60]=1
print(test_score)
