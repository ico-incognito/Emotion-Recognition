import numpy as np
import tensorflow as tf

#Control parameters
m_train = 99
m_test = 718

#Layer specifications
ip_h = 48
ip_w = 48

op = 7

filename1 = "fer2013/fer.csv"
filename2 = "fer2013/demo_test.csv"

#Read train set
x_train = []
y_train = []

with open(filename1) as inf:
	#Skip header
	next(inf)
	for line in inf:
		emotion, str_pixels, usage = line.strip().split(",")
		x1 = np.fromstring(str_pixels, sep = ' ', dtype = int)
		x_train = np.append(x_train, x1)
		y_train = np.append(y_train, emotion)
		print(emotion, x1)

y_train = y_train.astype(int)

x_train = np.reshape(x_train, newshape = (m_train, ip_h, ip_w, 1))
y_train = np.reshape(y_train, newshape = (m_train, 1))

#Convert labels to one hot
y_train = (np.arange((y_train.max()) + 1) == y_train).astype(int)

print(y_train)
print(x_train[0])