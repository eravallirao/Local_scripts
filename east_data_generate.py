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
		text_file_name = ""
		vinnu_final = ""
		for elem in tree.iter():
			if elem.tag == "name" or elem.tag == "xmin" or elem.tag == "ymin" or elem.tag == "xmax" or elem.tag == "ymax" or elem.tag == "path" or elem.tag == "text":
				if elem.tag == "path":
					vinnu = str(elem.text)
					vinnu = str(vinnu.split("\\")[-1])
					vinnu = str(vinnu.split(".")[0]) + ".txt"
					text_file_name = vinnu
				elif elem.tag == "xmin":
					vinnu1 = str(elem.text) + ","
					vinnu_final += vinnu
					f = open(text_file_name,"a")
					f.write(vinnu)
					f.close()
				elif elem.tag == "ymin":
					vinnu2 = str(elem.text) + ","
					f = open(text_file_name,"a")
					f.write(vinnu)
					f.close()
				elif elem.tag == "xmax":
					vinnu3 = str(elem.text) + ","
					f = open(text_file_name,"a")
					f.write(vinnu)
					f.close()
				elif elem.tag == "ymax":
					vinnu4 = str(elem.text) + ","
					f = open(text_file_name,"a")
					f.write(vinnu + str("0") + "\n")
					f.close()
				elif elem.tag == "text":
					vinnu4 = str(elem.text) + ","
					f = open(text_file_name,"a")
					f.write(vinnu + str("0") + "\n")
					f.close()
				else: 
				    print("none")
				# vinnu_sec = " ".join(vinnu_final)
			else:
				continue
			i = i + 1
		print(vinnu_final + "_" + "text")

		# text_file_name = 
		# f = open(text_file_name,"w+")
		# f.write("This is line %d\r\n" % (i+1))
		# f.close() 
# print(vinnu_sec)
