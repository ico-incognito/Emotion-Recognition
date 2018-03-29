import pandas as pd
import numpy as np

filename = "fer2013/demo_test.csv"
dataset = pd.read_csv(filename)
y = pd.get_dummies(dataset.iloc[:, 0])
x1 = pd.DataFrame(dataset.pixels.str.split(' ', 2304).tolist())

x1 = x1.astype(float)
x1 /= 255

print(y)
print(x1)
