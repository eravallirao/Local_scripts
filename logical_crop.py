from PIL import Image, ImageDraw
import cv2
import os
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())
''
images = args["image"]
image_path_output = '/Users/sukshi/Downloads/Voter_micro_validate'

c = 0

for fl in os.listdir(images):
	#print(fl)
	if fl == ".DS_Store" or fl == "_DS_Store":
		#print(fl)
		print("stupid files")
	else:
		try:
			images2 = os.path.join(images,fl)
			c = c + 1
			# im = Image.open(images2)
			# im_width, im_height = im.size
			# print('im.size', im.size)
			###################################################### Left half of the image
			# im = im.crop((0, 0, im_width/2,im_height))  # (left, upper, right, lower)-tuple.
			frame = cv2.imread(images2)
			height, width, channels = frame.shape

			# height_1 = height/4
			# width_0 = width/5
			# width_1 = width_0*3
			# height_0 = height/5
			# height_1 = height*3
			# width_0 = int(width/2)
			# height_0 = int(height/2)
			width_1 = (0.15)*width
			width_2 = (0.17)*width
			width_3 = (0.97)*width
			width_4 = (0.85)*width
			height_1 = (0.12)*width
			height_2 = (0.17)*width
			height_3 = (0.97)*width
			height_4 = (0.87)*width

			if c <= 700:
				fin_img = frame[int(height_1):int(height_4), int(width_1):int(width_4)]
			elif c <=1500 and c >= 701:
				fin_img = frame[int(height_2):int(height), int(width_2):int(width_4)]
			elif c <=2100 and c >= 1501:
				fin_img = frame[int(height_1):int(height_4), 0:int(width_4)]
			else:
				fin_img = frame[0:int(height_3), int(width_2):int(width_4)]




			# Logic for QR
			# width_0 = int((width)*(1.7/3))
			# height_0 = int((height)*(1.1/3))
			# width_1 = int(width)
			# height_1 = int((height)*(2.9/3))

			
			# fin_img = frame[int(height_0):int(height_1), int(width_0):int(width_1)]
			# image_name_output = fl

			cv_interpolation = cv2.INTER_LANCZOS4
			# cropped = cv2.resize(frame,(224,224))
			fl2 = fl.split(".")[0]
			

			# im.save(image_path_output + "/" + image_name_output)
			cropped = cv2.resize(fin_img, dsize=(224, 224), interpolation=cv_interpolation)
			# cropped = cv2.resize(frame,(224,224))
			# cv2.imwrite( str(fl) + ".jpg",cropped)
			cv2.imwrite( str(image_path_output) + "/" + "face_Cropped_" + str(fl2) + ".jpg",cropped)
			# print('im.size', im.size)
			# print('*** Program Ended ***')
		except:
			print("failed")
			continue





