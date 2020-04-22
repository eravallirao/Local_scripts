# import required classes
 
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import random
import cv2
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
		font = ImageFont.truetype('Times_New_Roman_2.ttf', size=18)
		# fl2 = fl.split("_")[1]
		# for i in range(1):
		# img = cv2.imread(images2)
		# height, width, channels = img.shape
		image = Image.open(images2)
		width, height = image.size
		print(width, height)
		draw = ImageDraw.Draw(image)
		print("hello")
		itr = itr + 1
		(x, y) = int(width*(.1)),int(height*(.7))
		message = (random.choice(list(open('1st.txt'))))
		color = 'rgb(255,0,0)' # black color
		draw.text((x,y), message, fill=color, font=font)

		(x, y) = int(width*(.1)),int(height*(.75))
		message = (random.choice(list(open('2nd.txt'))))
		color = 'rgb(255,0,0)' # black color
		draw.text((x,y), message, fill=color, font=font)

		(x, y) = int(width*(.1)),int(height*(.80))
		message = (random.choice(list(open('3rd.txt'))))
		color = 'rgb(255,0,0)' # black color
		draw.text((x,y), message, fill=color, font=font)

		(x, y) = int(width*(.1)),int(height*(.85))
		message = (random.choice(list(open('4th.txt'))))
		color = 'rgb(255,0,0)' # black color
		draw.text((x,y), message, fill=color, font=font)

		# (x, y) = (57,305)
		# message = (random.choice(list(open('local_4.txt'))))
		# color = 'rgb(68, 68, 68)' # black color
		# draw.text((x, y), message, fill=color, font=font)

		# (x, y) = (59, 365)
		# message = (random.choice(list(open('state_space.txt'))))
		# color = 'rgb(68, 68, 68)' # black color
		# draw.text((x, y), message, fill=color, font=font)


		# (x, y) = (49, 497)
		# name = (random.choice(list(open('last_line.txt'))))
		# color = 'rgb(68, 68, 68)' # white color
		# draw.text((x, y), name, fill=color, font=font)

		fl2 = fl.split(".")[0]
		image.save(str(fl2) + ".jpg" )
		print("no image")


			# if fl2 == "31" or fl2 == "32" or fl2 == "36":		 
			# starting position of the message		 
			# (x, y) = (75, 163)
			# message = (random.choice(list(open('name_1.txt'))))
			# color = 'rgb(68, 68, 68)' # black color
			# draw.text((x, y), message, fill=color, font=font)

			# (x, y) = (70, 242)
			# message = (random.choice(list(open('name_1.txt'))))
			# color = 'rgb(68, 68, 68)' # black color
			# draw.text((x, y), message, fill=color, font=font)

			# (x, y) = (73, 304)
			# message = (random.choice(list(open('name_1.txt'))))
			# color = 'rgb(68, 68, 68)' # black color
			# draw.text((x, y), message, fill=color, font=font)

			# (x, y) = (65, 375)
			# message = (random.choice(list(open('h_no.txt'))))
			# color = 'rgb(68, 68, 68)' # black color
			# draw.text((x, y), message, fill=color, font=font)

			# (x, y) = (75, 439)
			# message = (random.choice(list(open('local_4.txt'))))
			# color = 'rgb(68, 68, 68)' # black color
			# draw.text((x, y), message, fill=color, font=font)

			# (x, y) = (73, 498)
			# message = (random.choice(list(open('state_space.txt'))))
			# color = 'rgb(68, 68, 68)' # black color
			# draw.text((x, y), message, fill=color, font=font)


			# (x, y) = (80, 633)
			# name = (random.choice(list(open('last_line.txt'))))
			# color = 'rgb(68, 68, 68)' # white color
			# draw.text((x, y), name, fill=color, font=font)

			# fl2 = fl.split(".")[0]
			# image.save(str(fl2) + "_" + str(itr) + ".jpg" )
				# image.save(fl)
				# image.save('optimized.png', optimize=True, quality=20)
			# elif fl2 == "38":
			# 	(x, y) = (62, 77)
			# 	message = (random.choice(list(open('name_1.txt'))))
			# 	color = 'rgb(68, 68, 68)' # black color
			# 	draw.text((x, y), message, fill=color, font=font)

			# 	(x, y) = (63, 134)
			# 	message = (random.choice(list(open('name_1.txt'))))
			# 	color = 'rgb(68, 68, 68)' # black color
			# 	draw.text((x, y), message, fill=color, font=font)

			# 	(x, y) = (55, 193)
			# 	message = (random.choice(list(open('name_1.txt'))))
			# 	color = 'rgb(68, 68, 68)' # black color
			# 	draw.text((x, y), message, fill=color, font=font)

			# 	(x, y) = (53, 250)
			# 	message = (random.choice(list(open('h_no.txt'))))
			# 	color = 'rgb(68, 68, 68)' # black color
			# 	draw.text((x, y), message, fill=color, font=font)

			# 	(x, y) = (39, 301)
			# 	message = (random.choice(list(open('local_4.txt'))))
			# 	color = 'rgb(68, 68, 68)' # black color
			# 	draw.text((x, y), message, fill=color, font=font)

			# 	(x, y) = (38, 360)
			# 	message = (random.choice(list(open('state_space.txt'))))
			# 	color = 'rgb(68, 68, 68)' # black color
			# 	draw.text((x, y), message, fill=color, font=font)


			# 	(x, y) = (41, 491)
			# 	name = (random.choice(list(open('last_line.txt'))))
			# 	color = 'rgb(68, 68, 68)' # white color
			# 	draw.text((x, y), name, fill=color, font=font)

			# 	fl2 = fl.split(".")[0]
			# 	image.save(str(fl2) + "_" + str(itr) + ".jpg" )
			# 	print("no image")