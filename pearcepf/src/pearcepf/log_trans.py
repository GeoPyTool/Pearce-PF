import math

data = [
      [1, 3],
      [1, 1],
      [2.4, 2.4],
      [3, 1]
]

transformed_data = [[math.pow(10, x) for x in pair] for pair in data]

print(transformed_data)