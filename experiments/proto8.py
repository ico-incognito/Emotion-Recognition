import numpy as np
import tensorflow as tf

#Control parameters
m_train = 32298
m_test = 3589
num_epochs = 100
minibatch_size = 64
learning_rate = 0.0001

#Layer specifications
ip_h = ip_w = 48
fc = 1024
op = 7

filename1 = "fer2013/train_private.csv"
filename2 = "fer2013/public_test.csv"

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

x_train = np.reshape(x_train, newshape = (m_train, ip_h, ip_w, 1))
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

x_test = np.reshape(x_test, newshape = (m_test, ip_h, ip_w, 1))
y_test = np.reshape(y_test, newshape = (m_test, 1))

y_test = (np.arange((y_test.max()) + 1) == y_test).astype(int)


#Normalize
x_train /= 255
x_test /= 255


#Create placeholders
x = tf.placeholder(tf.float32, [None, ip_h, ip_w, 1], name = "x")
y = tf.placeholder(tf.float32, [None, op], name = "y")
keep_prob = tf.placeholder(tf.float32)

#Initialize weights
W1 = tf.get_variable("W1", [5, 5, 1, 64], initializer = tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", [5, 5, 64, 64], initializer = tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", [4, 4, 64, 128], initializer = tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", [3200, fc], initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("b4", [1, fc], initializer = tf.zeros_initializer())
W5 = tf.get_variable("W5", [fc, op], initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.get_variable("b5", [1, op], initializer = tf.zeros_initializer())

#Perform convolution, activation and pooling
Z1 = tf.nn.conv2d(x, W1, strides = [1, 1, 1, 1], padding = 'VALID')
A1 = tf.nn.relu(Z1)

#Local Response Normalization
A1 = tf.nn.lrn(A1, 4, bias = 1.0)

P1 = tf.nn.max_pool(A1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID')

Z2 = tf.nn.conv2d(P1, W2, strides = [1, 1, 1, 1], padding = 'VALID')
A2 = tf.nn.relu(Z2)
P2 = tf.nn.max_pool(A2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID')

Z3 = tf.nn.conv2d(P2, W3, strides = [1, 1, 1, 1], padding = 'VALID')
A3 = tf.nn.relu(Z3)
P3 = tf.nn.max_pool(A2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID')

#Regularization
P3 = tf.nn.dropout(A3, keep_prob)

#Convert from 5x5x128 to 3200x1 array
P3 = tf.contrib.layers.flatten(P3)

Z4 = tf.add(tf.matmul(P3, W4), b4)
A4 = tf.nn.relu(Z4)

Z5 = tf.add(tf.matmul(A4, W5), b5)

#Compute cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z5, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

costs = []
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#Start tf session
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(num_epochs):
		start = 0
		epoch_cost = 0
		num_minibatches = int(m_train/minibatch_size)

		for minibatch in range(num_minibatches):
			#Mini batch processing
			minibatch_X = x_train[start: (start + minibatch_size), :, :]
			minibatch_Y = y_train[start: (start + minibatch_size), :]
			start += minibatch_size
			_ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {x: minibatch_X, y: minibatch_Y, keep_prob: 0.3})
			epoch_cost += minibatch_cost / num_minibatches

		if epoch % 10 == 0:
			print("Cost after epoch %i: %f" % (epoch, epoch_cost))
		if epoch % 1 == 0:
			costs.append(epoch_cost)

	correct_prediction = tf.equal(tf.argmax(Z5, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#Display accuracy
	#print("Train Accuracy:", accuracy.eval({x: x_train, y: y_train, keep_prob: 1.0}))
	#print("Test Accuracy:", accuracy.eval({x: x_test, y: y_test, keep_prob: 1.0}))
	print((tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
	#saver.save(sess, 'model11/fyp11.ckpt')

	
	num_minibatches = int(m_train/minibatch_size)
	train_accuracy = 0
	start = 0

	for minibatch in range(num_minibatches):
		#Mini batch processing
		minibatch_X = x_train[start: (start + minibatch_size), :, :]
		minibatch_Y = y_train[start: (start + minibatch_size), :]
		start += minibatch_size
		train_accuracy = train_accuracy + accuracy.eval({x: minibatch_X, y: minibatch_Y, keep_prob: 1.0})
	print("Train Accuracy:", train_accuracy/num_minibatches)
	
	num_minibatches = int(m_test/minibatch_size)
	test_accuracy = 0
	start = 0

	for minibatch in range(num_minibatches):
		#Mini batch processing
		minibatch_X = x_test[start: (start + minibatch_size), :, :]
		minibatch_Y = y_test[start: (start + minibatch_size), :]
		start += minibatch_size
		test_accuracy = test_accuracy + accuracy.eval({x: minibatch_X, y: minibatch_Y, keep_prob: 1.0})
	print("Test Accuracy:", test_accuracy/num_minibatches)
	