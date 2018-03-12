from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView
import json
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import tensorflow as tf
from django.http import JsonResponse

#Layer specifications for tf
ip_h = ip_w = 48
fc = 1024
op = 7

#Load haar cascade file
face_cascade = cv2.CascadeClassifier('ocv/haarcascade_frontalface_default.xml')

#Function to be called once to create tf graph
def load_model():
	#Create graph
	mygraph = tf.Graph()
	with mygraph.as_default():
		#Create placeholder
		x_tensor = tf.placeholder(tf.float32, [None, ip_h, ip_w, 1], name = "x")

		#Declare weights and biases
		W1 = tf.get_variable("W1", [5, 5, 1, 64])
		W2 = tf.get_variable("W2", [5, 5, 64, 64])
		W3 = tf.get_variable("W3", [4, 4, 64, 128])
		W4 = tf.get_variable("W4", [3200, fc])
		b4 = tf.get_variable("b4", [1, fc])
		W5 = tf.get_variable("W5", [fc, op])
		b5 = tf.get_variable("b5", [1, op])

		#Perform convolution, activation and pooling
		Z1 = tf.nn.conv2d(x_tensor, W1, strides = [1, 1, 1, 1], padding = 'VALID')
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
		
	return x_tensor, init, saver, Z5, mygraph


#Function to be called in real time
def display_result(x_tensor, x_ip, init, saver, Z5, mygraph):
	#Convert x_ip to correct format
	x_ip = np.reshape(x_ip, newshape = (1, ip_h, ip_w, 1))
	x_ip = x_ip.astype(float)
	x_ip /= 255
	
	#Start tf session
	sess = tf.InteractiveSession(graph = mygraph)
	sess.run(init)

	#Restore weights and biases
	saver.restore(sess, "ocv/model10/fyp10.ckpt")

	#Calculate probability of emotion
	emotion = tf.nn.softmax(Z5)
	result = tf.argmax(emotion, 1)
	softmax = emotion.eval({x_tensor: x_ip})
	sess.close()

	#Variable for each emotion probability
	anger = softmax[0, 0]
	disgust = softmax[0, 1]
	fear = softmax[0, 2]
	happy = softmax[0, 3]
	sad = softmax[0, 4]
	surprise = softmax[0, 5]
	neutral = softmax[0, 6]

	return anger, disgust, fear, happy, sad, surprise, neutral

@csrf_exempt
def index(request):
	#Load tf model
	global x_tensor, init, saver, Z5, mygraph
	x_tensor, init, saver, Z5, mygraph = load_model()

	#Display initial page
	return render(request, 'ocv/index.html')

@csrf_exempt
def opencv(request):
	#Catch POST request
	frame1 = request.POST.get("frame")

	#Convert frame1 to correct format
	frame = np.fromstring(frame1, sep = ',', dtype = np.uint8)
	frame = np.reshape(frame, newshape = (144, 192, 4))

	#RGBA to Gray
	gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

	#Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	#Faces not detected
	if not len(faces) > 0:
		anger = disgust = fear = happy = sad = surprise = neutral = np.float64(0)
	
	else:
	#Detect face rectangles
		for (x,y,w,h) in faces:
			continue

		#Capture face
		face = gray[y-0 : y+h+0, x-0 : x+w+0]

		#Resize for input to CNN
		face = cv2.resize(face, (48, 48))

		#Pass image to CNN
		anger, disgust, fear, happy, sad, surprise, neutral = display_result(x_tensor, face, init, saver, Z5, mygraph)

		#Make variables JSON serializable ie float32 to float64
		anger = anger.astype(np.float64)
		disgust = disgust.astype(np.float64)
		fear = fear.astype(np.float64)
		happy = happy.astype(np.float64)
		sad = sad.astype(np.float64)
		surprise = surprise.astype(np.float64)
		neutral = neutral.astype(np.float64)

	#Send back response
	return JsonResponse({"anger" : anger,
		"disgust" : disgust,
		"fear" : fear,
		"happy" : happy,
		"sad" : sad,
		"surprise" : surprise,
		"neutral" : neutral})