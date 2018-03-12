import numpy as np
import tensorflow as tf

#Control parameters
m_train = 2870
m_test = 718
num_epochs = 200
minibatch_size = 64
learning_rate = 0.0001

#No of nodes in each layer
ip = 2304
l1 = 500
l2 = 350
l3 = 200
l4 = 90
op = 7

filename1 = "fer2013/demo_train.csv"
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

y_train = y_train.astype(int)

x_train = np.reshape(x_train, newshape = (m_train, ip))
y_train = np.reshape(y_train, newshape = (m_train, 1))

#Convert labels to one hot
y_train = (np.arange((y_train.max()) + 1) == y_train).astype(int)


#Read test set
x_test = []
y_test = []

with open(filename2) as inf:
	next(inf)
	for line in inf:
		emotion, str_pixels, usage = line.strip().split(",")
		x1 = np.fromstring(str_pixels, sep = ' ', dtype = int)
		x_test = np.append(x_test, x1)
		y_test = np.append(y_test, emotion)

y_test = y_test.astype(int)

x_test = np.reshape(x_test, newshape = (m_test, ip))
y_test = np.reshape(y_test, newshape = (m_test, 1))

y_test = (np.arange((y_test.max()) + 1) == y_test).astype(int)


#Create placeholders
x = tf.placeholder(tf.float32, [None, ip], name = "x")
y = tf.placeholder(tf.float32, [None, op], name = "y")

#Initialize weights and biases
W1 = tf.get_variable("W1", [ip, l1], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", [1, l1], initializer = tf.zeros_initializer())

W2 = tf.get_variable("W2", [l1, l2], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", [1, l2], initializer = tf.zeros_initializer())

W3 = tf.get_variable("W3", [l2, l3], initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("b3", [1, l3], initializer = tf.zeros_initializer())

W4 = tf.get_variable("W4", [l3, l4], initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("b4", [1, l4], initializer = tf.zeros_initializer())

W5 = tf.get_variable("W5", [l4, op], initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.get_variable("b5", [1, op], initializer = tf.zeros_initializer())

#Perform activations
Z1 = tf.add(tf.matmul(x, W1), b1)
A1 = tf.nn.relu(Z1)

Z2 = tf.add(tf.matmul(A1, W2), b2)
A2 = tf.nn.relu(Z2)

Z3 = tf.add(tf.matmul(A2, W3), b3)
A3 = tf.nn.relu(Z3)

Z4 = tf.add(tf.matmul(A3, W4), b4)
A4 = tf.nn.relu(Z4)

Z5 = tf.add(tf.matmul(A4, W5), b5)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z5, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

costs = []
init = tf.global_variables_initializer()

#Start tf session
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(num_epochs):
		start = 0
		epoch_cost = 0
		num_minibatches = int(m_train/minibatch_size)

		for minibatch in range(num_minibatches):
			#Mini batch processing
			minibatch_X = x_train[start: (start + minibatch_size), 0:]
			minibatch_Y = y_train[start: (start + minibatch_size), 0:]
			start += minibatch_size
			_ , minibatch_cost = sess.run([optimizer, cost], feed_dict={x: minibatch_X, y: minibatch_Y})
			epoch_cost += minibatch_cost / num_minibatches

		if epoch % 10 == 0:
			print("Cost after epoch %i: %f" % (epoch, epoch_cost))
		if epoch % 5 == 0:
			costs.append(epoch_cost)

	correct_prediction = tf.equal(tf.argmax(Z5, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#Display accuracy
	print("Train Accuracy:", accuracy.eval({x: x_train, y: y_train}))
	print("Test Accuracy:", accuracy.eval({x: x_test, y: y_test}))

