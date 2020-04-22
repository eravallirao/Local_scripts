import os
import argparse
import dlib
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())


images = args["image"]	


for fl in os.listdir(images):
	print(fl)
	if fl == ".DS_Store" or fl == "_DS_Store":
		#print(fl)
		print("stupid files")

	else:
		images2 = os.path.join(images,fl)	
		frame = cv2.imread(images2)
		detector = dlib.get_frontal_face_detector()
		try:
			if len(frame) > 0:
				#frame2 = imutils.resize(frame, width=450)
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				detected_faces = detector(gray, 0)
				
				height = frame.shape[0]
				width = frame.shape[1]
				height2 = gray.shape[0]

				face_rect = detected_faces[0]
					
				a = face_rect.left()
				b = face_rect.top()
				c = face_rect.right()
				d = face_rect.bottom()
				#print(a,b,c,d)

				if(face_rect.left() < 0):
					a = 0
				if(face_rect.top() < 0):
					b = 0
				if(face_rect.right() > width):
					c = width-1
				if(face_rect.bottom() > height):
					d = height-1

				#print(a,b,c,d)

				face = frame[b:d , a:c]
				fl2 = fl.split(".")[0]
				# cv2.imwrite( "cropped" + ".jpg",face)

				# cropped = cv2.resize(face,(300,300))
				# cv2.imwrite( str(fl) + ".jpg",cropped)
				cv2.imwrite( str(images) + "/" + str(fl2) + "_crop.jpg",face)

				# ratio = height*width/((c-a)*(d-b))
				# print(ratio)

				# a1 = int(a*width/450)
				# b1 = int(b*height/height2)
				# c1 = int(c*width/450)
				# d1 = int(d*height/height2)

				# face2 = frame[b1:d1 , a1:c1]
				# cv2.imwrite( "cropped1" + ".jpg",face2)
		except:
			os.remove(images2)
			print("not cropped, so deleting:", fl)
			continue




			

		