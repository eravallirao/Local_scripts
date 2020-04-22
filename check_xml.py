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
			# print(elem.tag)
			tagss.append(elem.tag)
			print(elem.text)
		print("*************************************************")
# print(to_be_removed)
# print(vinnu_sec)
