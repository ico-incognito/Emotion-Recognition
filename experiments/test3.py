#Code to read dataset into np array and convert labels to one hot
import numpy as np

filename = "fer2013/fer.csv"

x = []
y = []

with open(filename) as inf:
	# Skip header
	next(inf)
	for line in inf:
		# Read data
		emotion, str_pixels, usage = line.strip().split(",")
		x1 = np.fromstring(str_pixels, sep=' ', dtype=int)
		x = np.append(x, x1)

		y = np.append(y, emotion)
		# Print
		print(emotion, x1)

y= y.astype(int)

x = np.reshape(x, newshape=(99,2304))
y = np.reshape(y, newshape=(99,1))

y = (np.arange((y.max())+1) == y).astype(int)
print(y)
print(x)