import math

# data = [
#       [1, 3],
#       [1, 1],
#       [2.4, 2.4],
#       [3, 1]
# ]

# transformed_data = [[math.pow(10, x) for x in pair] for pair in data]

# print(transformed_data)

# 定义两个点
point1 = [20,100]
point2 = [2,3]

# 计算斜率
slope = (point2[1] - point1[1]) / (point2[0] - point1[0])

# 计算y截距
intercept = point1[1] - slope * point1[0]

# 计算x=1000时的y值
x =23.30612244897959
y = slope * x + intercept

print(y)