# from PIL import Image

# img = Image.open('res96.jpg').convert('L')  # convert image to 8-bit grayscale
# WIDTH, HEIGHT = img.size

# data = list(img.getdata())
# print(data)
# count = 0
# for i in data:
#     # print (i)
#     count = count + 1
# print(count)
import numpy as np
from PIL import Image

import os
import argparse
import dlib
import cv2
import imutils
import csv
# from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())


images = args["image"]
my_list1 = []	
count = 0


for fl in os.listdir(images):
	#print(fl)
	if fl == ".DS_Store" or fl == "_DS_Store":
		#print(fl)
		print("stupid files")

	else:
		images2 = os.path.join(images,fl)
		# print(fl)
		# img = Image.open(images2).convert('L')
		# data = str(list(img.getdata()))

		# images_g = np.array([to_grayscale(images2[i]) for i in range(images2.shape[0])])

		# # np_im = numpy.array(img)
		# print(images_g)


		img_file = Image.open(images2) # imgfile.show()

		# get original image parameters...
		width, height = img_file.size
		format = img_file.format
		mode = img_file.mode

		# Make image Greyscale
		img_grey = img_file.convert('L')
		#img_grey.save('result.png')
		#img_grey.show()

		# Save Greyscale values
		value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
		# value = value.flatten()
		print(value)
		with open("img_pixels_12356.csv", 'a') as f:
			fieldnames = ['x']
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writerow({'x' :value})



		# with open("devuda_1.csv", 'a', newline='') as csvfile:
		# 	fieldnames = ['x']
		# 	print("printinfcsv")
		# 	# writer.writeheader()
		# 	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		# 	writer.writerow({'x': my_list})










		# print (str(data)[1:-1])
		# my_list = str(data)[1:-1]
		# # my_str_as_bytes = str.encode(my_list)
		# # # print(my_str_as_bytes)
		# # # mine1 = (" ".join(map(str,my_str_as_bytes)))
		# # print(my_str_as_bytes)

		# # img = Image.open(images2)
		# # arr = numpy.array(img)
		# my_list1.append(my_list)
		# print(my_list + " " + ":")
		# mine = *my_list1
# mine1 = (" ".join(map(str,my_list1)))

		# csvfile = 'Ribhu_land_1_1.csv'
		# with open(csvfile, 'r') as fin, open('new_'+csvfile, 'w') as fout:
		#     reader = csv.reader(fin, lineterminator='\n')
		#     writer = csv.writer(fout, lineterminator='\n')
		#     # if you_have_headers:
		#     #     writer.writerow(next(reader) + ["New_Image"])
		#     for row, val in zip(reader, my_list):
		#         writer.writerow(row + [my_list])
		#         count = count + 1

		# chootu = str(data).translate(str.maketrans('' , ''), '[]\'')
		# print(chootu)
		# print(data.split["["])
		# with open("devuda_1.csv", 'a', newline='') as csvfile:
		# 	fieldnames = ['x']
		# 	print("printinfcsv")
		# 	# writer.writeheader()
		# 	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		# 	writer.writerow({'x': my_list})



# from pandas import DataFrame
# # C = {'Programming language': [my_list1],
# #     }
# df = DataFrame(my_list1, columns= ['Image'])
# export_csv = df.to_csv (r'Something_1.csv', index = None, header=True) # here you have to write path, where result file will be stored
# print (df)
