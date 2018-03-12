import numpy as np

filename = "fer2013/fer.csv"
y = np.genfromtxt(filename, delimiter=',', usecols=0, unpack=True, dtype=int, skip_header=1)
y = np.reshape(y, newshape=(None,1))
x_str = np.genfromtxt(filename, delimiter=',', usecols=1, unpack=True, dtype=np.string_, skip_header=1)

x = np.empty([None, 2304], dtype= int)
x = np.fromstring(x_str, sep=' ', dtype= int)
#x = np.reshape(x, newshape=(99,2304))


print(x_str)
#print(x_str.dtype)
#print(x_str.shape)

print(y)
print(y.dtype)
print(y.shape)

print(x)
print(x.dtype)
print(x.shape)