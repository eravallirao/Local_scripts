from PIL import Image, ImageDraw
import cv2
import os
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())

images = args["image"]
image_path_output = '/Users/sukshi/Downloads'


for fl in os.listdir(images):
	#print(fl)
	if fl == ".DS_Store" or fl == "_DS_Store":
		#print(fl)
		print("stupid files")
	else:
		images2 = os.path.join(images,fl)
		#a = im_height 
		#b = im_width/2 
		# im = Image.open(images2)
		# im_width, im_height = im.size
		# print('im.size', im.size)
		###################################################### Left half of the image
		# im = im.crop((0, 0, im_width/2,im_height))  # (left, upper, right, lower)-tuple.
		frame = cv2.imread(images2)
		height = frame.shape[0]
		width = frame.shape[1]
		h0 = int(0.6*(height))
		h1 = int(0.9*(height))
		w0 = int(0.1*(width))
		w1 = int(0.9*(width))
		# im_width = int(width/2)
		# print(height,width,im_width)
		
		fin_img = frame[h0:h1, w0:w1]
		print(fin_img)
		# image_name_output = fl

		cv_interpolation = cv2.INTER_LANCZOS4
		# cropped = cv2.resize(frame,(224,224))
		fl2 = fl.split(".")[0]
		

		# im.save(image_path_output + "/" + image_name_output)
		cropped = cv2.resize(fin_img, dsize=(224, 224), interpolation=cv_interpolation)
		# cropped = cv2.resize(frame,(224,224))
		# cv2.imwrite( str(fl) + ".jpg",cropped)
		cv2.imwrite(str(image_path_output) + "/" + "face_Cropped_" + str(fl2) + ".jpg", cropped)
		# print('im.size', im.size)
		# print('*** Program Ended ***')





