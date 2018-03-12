import tensorflow as tf
import numpy as np

filename = "fer2013/fer.csv"

emotion = tf.placeholder(tf.int32, name='emotion')
str_pixels = tf.placeholder(tf.string, name='str_pixels')
x = tf.placeholder(tf.int32, shape=[None,2304], name='x')
x1 = tf.placeholder(tf.int32, shape=[None,2304], name='x1')
usage = tf.placeholder(tf.string, name='usage')
y = tf.placeholder(tf.int32, shape=[None, 7], name='y')

#printerop = tf.Print(pixels, [emotion, pixels, usage], name='printer')

with tf.Session() as sess:
	sess.run( tf.global_variables_initializer())
	with open(filename) as inf:
	# Skip header
		next(inf)
		for line in inf:
			# Read data, using python, into our features
			emotion, str_pixels, usage = line.strip().split(",")
			x1 = np.fromstring(str_pixels, sep=' ', dtype=int)
			x = np.append(x, x1, axis=None)


			y = np.append(y, emotion, axis=None)
			# Run the Print ob
			print(emotion, x1)

y = tf.cast(y, tf.int32)
y = tf.one_hot(y, 7)
y = np.reshape(y, newshape=(100,7))
#x = np.reshape(x, newshape=(100,2304))
print(y)
print(x)