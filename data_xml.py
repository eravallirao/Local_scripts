import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

import argparse


# it takes the folder name of xml files as argument

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())


images = args["image"]	

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
		i = 0
		vinnu_final = ""
		for elem in tree.iter():
			if elem.tag == "name" or elem.tag == "xmin" or elem.tag == "ymin" or elem.tag == "xmax" or elem.tag == "ymax" or elem.tag == "path":
				if elem.tag == "path":
					vinnu = str(elem.text)
					# fl2 = fl.split(".")[0]
					vinnu = "C:" + str(vinnu.split("\\")[-1])
					vinnu_final += vinnu
					# print(vinnu)
				else:
					vinnu = str(elem.text) + ","
					vinnu_final += vinnu
				# vinnu_sec = " ".join(vinnu_final)
			else:
				continue
				# # print(str(elem.tag) + "__" + str(elem.text))
				# print("escape")
				# if elem.text == " ":
				# 	print("deepa")
				# else:
				# 	vinnu = str(elem.text) + ","
				# 	vinnu_final += vinnu
				# 	vinnu_sec = " ".join(vinnu_final)
			# print(elem.text)
			i = i + 1
		print(vinnu_final + "_" + "text")
# print(vinnu_sec)
