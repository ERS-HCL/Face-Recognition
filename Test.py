# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import h5py
import _pickle as cPickle
import os
from imutils import paths
import glob
import datetime
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())


model = cPickle.load(open("data/classifier.cpickle","rb"))

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
#print "enter the person name"
#name = raw_input()
#name = input("enter the person name ")

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
 
# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])
#features=[]
#labels=[]
#folder_name="db/"
#consec = None
#book = Workbook()
#sheet = book.active

#prev_pred = "0"
while True:
	# grab the current frame
	(grabbed, image) = camera.read()
 
	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 	# detect faces in the grayscale image
	rects = detector(gray, 1)
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		face_descriptor = facerec.compute_face_descriptor(image, shape)
		face1= np.array(face_descriptor)
		
		face2=face1.reshape(1,-1)
		print(face2.shape)
		#face1=face_descriptor.reshape(-1,1)
		prediction = model.predict(face2)[0]
		"""if consec is None:
			consec = [prediction, 1]
			#color = (0, 255, 0)
 
		# if predicted face matches the name in the consecutive list, then update the
		# total count
		elif prediction == consec[0]:
			consec[1] += 1"""
 
		# if the prediction has been "unknown" for a sufficient number of frames,
		# then we have an intruder
		
			
		pred = model.predict_proba(face2)[0]
		#prediction        = classifier.predict_proba(flat)
		# grab the first numpy array
		#predicted_probab  = prediction[0]
			
		# print the maximum probability obtained
		print((np.max(pred)))

		print(prediction)
		if np.max(pred) > 0.20:
			label_text = prediction.decode("utf-8")	
			#if label_text ==

			#fil=os.path.join(folder_name,label_text)
			#sam=glob.glob(fil + "/*.jpg")
			finaa= label_text
			#print(finaa.decode("utf-8"))
			print(finaa)

			#im=cv2.imread(sam[0])
			#cv2.putText(im, "{}".format(finaa), (10, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			#cv2.imshow(finaa,im)
			
			
		else:
			label_text = "unknown"
		#print pred
		#featu=face_descriptor.rQeshape(1,-1)
		#print featu.shape
		#print()
		#feat=np.reshape(face_descriptor, (1, 128))
		
		#features.append(face_descriptor)
		#labels.append(name)
		#print feat.shape
		#print feat
		#cv2.waitKey(100)
		
		#prediction = model.predict(face_descriptor)[0]
		# grab the first numpy array
		#predicted_probab  = prediction[0]
	
		#print prediction
		#cv2.waitKey(0)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		cv2.putText(image, "{}".format(label_text), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# anad draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

		# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", image)
	
	#cv2.destroyWindow("matched image")
			
	#cv2.waitKey(0)
	# show the frame to our screen
	#cv2.imshow("Frame", imutils.resize(frame, width=600))
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

#book.save("attendance.xlsx")
#print features.shape
#cv2.waitKey(100)

 
# clean up the camera and close any open windows
camera.release()
cv2.destroyAllWindows()




