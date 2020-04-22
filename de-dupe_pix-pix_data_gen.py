# import required classes
 
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import random
import cv2
import numpy as np
# print(random.choice(list(open('file.txt'))))

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())


images = args["image"]
itr = 0
for fl in os.listdir(images):
	#print(fl)
	if fl == ".DS_Store" or fl == "_DS_Store":
		#print(fl)
		print("stupid files")
	else:
		images2 = os.path.join(images,fl)
		# Create a black image
		image = Image.open(images2)
		draw = ImageDraw.Draw(image)
		img = cv2.imread(images2)
		height, width, channels = img.shape

		# cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# pil_im = Image.fromarray(cv2_im_rgb)
		# draw = ImageDraw.Draw(pil_im)
		color = 'rgb(68, 68, 68)' # black color
		message = (random.choice(list(open('1st.txt'))))
		print(message)
		font = ImageFont.truetype("Times_New_Roman_2.ttf", 50)
		draw.text((int(width*2),int(height*.7)), str(message), fill=color, font=font)

		message = (random.choice(list(open('2nd.txt'))))
		print(message)
		font = ImageFont.truetype("Times_New_Roman_2.ttf", 50)
		draw.text((int(width*2),int(height*.75)), str(message), fill=color, font=font)

		message = (random.choice(list(open('3rd.txt'))))
		print(message)
		font = ImageFont.truetype("Times_New_Roman_2.ttf", 50)
		draw.text((int(height*.80),int(width*2)), str(message), fill=color, font=font)

		message = (random.choice(list(open('4th.txt'))))
		print(message)
		font = ImageFont.truetype("Times_New_Roman_2.ttf", 50)
		draw.text((int(height*.85),int(width*2)), str(message), fill=color, font=font)


		fl2 = fl.split(".")[0]
		image.save(str(fl2) + ".jpg" )
		# cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
		# cv2.imwrite(str(fl2) + ".jpg", cv2_im_processed)

