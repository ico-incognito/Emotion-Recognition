import numpy as np
import tensorflow as tf
import cv2

#Layer specifications
ip_h = ip_w = 48
fc = 1024
op = 7
filename1 = "fer2013/public_test.csv"

#Function to be called once
def load_model():
	#Create placeholders
	x = tf.placeholder(tf.float32, [None, ip_h, ip_w, 1], name = "x")

	#Declare weights and biases
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


#Function to be called in real time
def display_result(x, x_ip, init, saver, Z5, y):
	x_draw = np.reshape(x_ip, newshape = (ip_h, ip_w))
	x_draw = x_draw.astype(np.uint8)
	x_draw = cv2.resize(x_draw, (192, 144))

	x_ip = np.reshape(x_ip, newshape = (1, ip_h, ip_w, 1))
	x_ip /= 255
	
	#Start tf session
	sess = tf.InteractiveSession()
	sess.run(init)

	#Restore weights and biases
	saver.restore(sess, "model10/fyp10.ckpt")

	#Calculate probability of emotion
	softmax = tf.nn.softmax(Z5)
	#emotion = tf.argmax(softmax, 1)
	prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
	if prediction.eval({x: x_ip}):
		cv2.imwrite("correct/face-" + str(count) + ".jpg", x_draw)
	else:
		cv2.imwrite("incorrect/face-" + str(count) + ".jpg", x_draw)
	sess.close()


x, init, saver, Z5 = load_model()
x_ip = []
count = 1

with open(filename1) as inf:
	next(inf)
	for line in inf:
		emotion, str_pixels, usage = line.strip().split(",")
		x_ip = np.fromstring(str_pixels, sep = ' ', dtype = float)

		emotion = int(emotion)
		y = np.zeros((1, 7))
		y[0, emotion] = 1

		display_result(x, x_ip, init, saver, Z5, y)
		count += 1