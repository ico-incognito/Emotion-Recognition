import numpy as np
import tensorflow as tf

ip_h = ip_w = 48
op = 7

#Create placeholders
x = tf.placeholder(tf.float32, [None, ip_h, ip_w, 1], name = "x")
y = tf.placeholder(tf.float32, [None, op], name = "y")
keep_prob = tf.placeholder(tf.float32)

#Initialize weights
W1 = tf.get_variable("W1", [5, 5, 1, 64], initializer = tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", [5, 5, 64, 64], initializer = tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", [4, 4, 64, 128], initializer = tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", [3200, 3072], initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("b4", [1, 3072], initializer = tf.zeros_initializer())
W5 = tf.get_variable("W5", [3072, op], initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.get_variable("b5", [1, op], initializer = tf.zeros_initializer())

saver = tf.train.Saver()

#Start tf session
with tf.Session() as sess:
	saver.restore(sess, "models/fyp.ckpt")
	print("W:", W1.eval())
	print((tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
