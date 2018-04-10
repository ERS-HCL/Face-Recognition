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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

#model = cPickle.loads(open("classifier.cpickle").read())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
number=0;
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
print("enter the person name")
name = input()
folder_name="data/"
#name = input("enter the person name ")
#dir = os.path.join(train_path, training_name)
fil=name+"_featrures.h5"
#fil=os.path.join(name,"_featrures.h5")
fil1=os.path.join(folder_name,fil)

print(fil1)

fil2=name+"_labels.h5"
fil3=os.path.join(folder_name,fil2)

print(fil3)

#if not os.path.exists(fil):
	#os.makedirs(fil)

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
 
# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])
features=[]
labels=[]
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
		#(x, y, w, h) = max(rect)
		#x, y, w, h = [ v for v in rect ]
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		#print rect.dtype
		cro=image[y: y + h, x: x + w]
		cv2.imshow("cropped image",cro)
		#fram= os.path.join(fil+"/",str(number)+ "." + "jpg")
		number+=1
		#print fram
		#cv2.imwrite(fram,cro)
		cv2.waitKey(1)
		face_descriptor = facerec.compute_face_descriptor(image, shape)
		#featu=face_descriptor.reshape(1,-1)
		#print featu.shape
		#print()
		#feat=np.reshape(face_descriptor, (1, 128))
		
		features.append(face_descriptor)
		labels.append(name)
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
		
		
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		#cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# anad draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

		# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", image)
	#cv2.waitKey(0)
	# show the frame to our screen
	#cv2.imshow("Frame", imutils.resize(frame, width=600))
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break


#print features.shape
#cv2.waitKey(100)
h5f_data=h5py.File(fil1,'a')
h5f_label=h5py.File(fil3,'a')
h5f_data.create_dataset('dataset_1',data=np.array(features))
fea=np.array(features)
print(fea.shape)
cv2.waitKey(100)
h5f_label.create_dataset('dataset_1',data=np.array(labels).astype('|S10'))
h5f_data.close()
h5f_label.close()

 
# clean up the camera and close any open windows
camera.release()
cv2.destroyAllWindows()




