import numpy as np
import tensorflow as tf

#Layer specifications
ip_h = ip_w = 48
fc = 1024
op = 7

filename1 = "fer2013/appl.csv"


def load_model():
	#Create placeholders
	x = tf.placeholder(tf.float32, [None, ip_h, ip_w, 1], name = "x")

	#Initialize weights
	W1 = tf.get_variable("W1", [5, 5, 1, 64])
	W2 = tf.get_variable("W2", [5, 5, 64, 64])
	W3 = tf.get_variable("W3", [4, 4, 64, 128])
	W4 = tf.get_variable("W4", [3200, fc])
	b4 = tf.get_variable("b4", [1, fc])
	W5 = tf.get_variable("W5", [fc, op])
	b5 = tf.get_variable("b5", [1, op])

	#Perform convolution, activation and pooling
	Z1 = tf.nn.conv2d(x, W1, strides = [1, 1, 1, 1], padding = 'VALID')
	A1 = tf.nn.relu(Z1)
	A1 = tf.nn.lrn(A1, 4, bias = 1.0)
	P1 = tf.nn.max_pool(A1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID')

	Z2 = tf.nn.conv2d(P1, W2, strides = [1, 1, 1, 1], padding = 'VALID')
	A2 = tf.nn.relu(Z2)
	P2 = tf.nn.max_pool(A2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID')

	Z3 = tf.nn.conv2d(P2, W3, strides = [1, 1, 1, 1], padding = 'VALID')
	A3 = tf.nn.relu(Z3)

	P3 = tf.contrib.layers.flatten(A3)

	Z4 = tf.add(tf.matmul(P3, W4), b4)
	A4 = tf.nn.relu(Z4)

	Z5 = tf.add(tf.matmul(A4, W5), b5)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	return x, init, saver, Z5

def display_result(x, x_ip, init, saver, Z5):
	x_ip = np.reshape(x_ip, newshape = (1, ip_h, ip_w, 1))
	x_ip /= 255
	#Start tf session
	sess = tf.InteractiveSession()
	sess.run(init)
	saver.restore(sess, "model10/fyp10.ckpt")
	emotion = tf.nn.softmax(Z5)
	result = tf.argmax(emotion, 1)

	print("Softmax:", emotion.eval({x: x_ip}))
	print("Result:", result.eval({x: x_ip}))

	sess.close()


x, init, saver, Z5 = load_model()
x_ip = []

with open(filename1) as inf:
	for line in inf:
		emotion, str_pixels, usage = line.strip().split(",")
		x_ip = np.fromstring(str_pixels, sep = ' ', dtype = float)

display_result(x, x_ip, init, saver, Z5)
display_result(x, x_ip, init, saver, Z5)
display_result(x, x_ip, init, saver, Z5)
