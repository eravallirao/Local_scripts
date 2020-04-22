import argparse
import cv2
import imutils
import os
from PIL import Image
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',type=str,default='',
	help="path")
args=vars(ap.parse_args())

path=os.getcwd()

# size=(224,224)

img_path=args["image"]
imgs_path=os.listdir(img_path)
for images in imgs_path:
	if images == ".DS_Store":
		print("false")
	else:
		try:	
			print(str(images))
			input_images=os.path.join(img_path,images)
			#print(input_images)
			img=cv2.imread(input_images)
			#im = Image.open(input_images)
			# cropped=image[0:0,224,150]
			img_name = images.split(".")[0]
			print(img_name)
			height=img.shape[0]
			width=img.shape[1]
			# print(width)
			# h1=int(8*height/10)
			h2=int(height/4)
			# print(h1)
			h=int(height/3)
			h1=height-h
			# he=int(height/10)
			# h2=height*he


			fin_img = img[h2:h1, 0:width]
			cropped = cv2.resize(fin_img,(224,224))
			# final=np.asarray(fin_img)
			# # # fin_img=im.resize(size,resample=0)
			cv2.imwrite(path+"/"+str(img_name)+".jpg",cropped)
		except:
			os.remove(input_images)
	# fin_img.save(images)