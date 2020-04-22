import os
import glob
import pandas as pd
import csv
import xml.etree.ElementTree as ET

import argparse


# it takes the folder name of xml files as argument

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())


images = args["image"]
tagss = []
to_be_removed = []

# /mnt/Train_YOLO/tensorflow-yolov3/data/dataset/train
for fl in os.listdir(images):
	#print(fl)
	if fl == ".DS_Store" or fl == "_DS_Store":
		print(fl)
		# print("stupid files")

	else:
		images2 = os.path.join(images,fl)
		# print(images2)
		# import xml.etree.ElementTree as ET
		tree = ET.parse(images2)
		# print(tree)
		i = 0
		vinnu_final = ""
		for elem in tree.iter():
			# print(elem.text)
			# print(elem.tag)
			tagss.append(elem.tag)
			# filename,width,height,class,xmin,ymin,xmax,ymax


			if elem.tag == "filename" or elem.tag == "width" or elem.tag == "height" or elem.tag == "name" or elem.tag == "xmin" or elem.tag == "ymin" or elem.tag == "xmax" or elem.tag == "ymax":
				if elem.tag == "filename":
					filename_1 = elem.text
				elif elem.tag == "width":
					width_1 = elem.text
				elif elem.tag == "height":
					height_1 = elem.text
				elif elem.tag == "name":
					class_1 = elem.text
					print(class_1)
				elif elem.tag == "xmin":
					xmin_1 = elem.text
				elif elem.tag == "ymin":
					ymin_1 = elem.text
				elif elem.tag == "xmax":
					xmax_1 = elem.text
				elif elem.tag == "ymax":
					ymax_1 = elem.text
				else:
					print("done")
			else:
				continue
		with open("csv_test.csv", 'a', newline='') as csvfile:
			fieldnames = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
			# writer.writeheader()
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writerow({'filename': filename_1, 'width': width_1, 'height': height_1, 'class': class_1, 'xmin': xmin_1, 'ymin': ymin_1, 'xmax': xmax_1, 'ymax': ymax_1})
