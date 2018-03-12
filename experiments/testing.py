import numpy as np
import cv2
import tensorflow as tf
from tens import load_model
from tens import display_result

x_tensor, init, saver, Z5 = load_model()
x_ip = []

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)

while 1:
	ret, img = cap.read()
	'''print(img)
	print(img.shape)
	print(img.dtype)'''
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	'''print(gray)
	print(gray.shape)
	print(gray.dtype)'''
	#gray = cv2.resize(gray, (192, 144))
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)


	#	cv2.rectangle(gray,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
		#roi_gray = gray[y:y+h, x:x+w]
		#roi_color = img[y:y+h, x:x+w]
	if not len(faces) > 0:
		continue
	else:
		for (x,y,w,h) in faces:
			continue
		face = gray[y-0 : y+h+0, x-0 : x+w+0]
		#face = gray[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
		cv2.imshow('img', face)
		face = cv2.resize(face, (48, 48))
		display_result(x_tensor, face, init, saver, Z5)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()