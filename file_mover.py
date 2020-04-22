import shutil
import os
import csv
import argparse


ap=argparse.ArgumentParser()
ap.add_argument('-f','--file',type=str,default='',
	help="path to file")
ap.add_argument('-s','--source', type=str, default='true',
 help="path to source folder")
ap.add_argument('-d','--destination', type=str, default='true',
 help="path to destination folder")
args=vars(ap.parse_args())

file=args["file"]
src = args["source"]
dst = args["destination"]


with open(file) as file:
	lines = file.readlines()
	for file_name in lines:
		try:
			print(file_name)
			file_name = file_name
			print(file_name)
			source=src+'/'+file_name
			print(source)
			# print(source)
			shutil.move(source[:-1], dst)
			# os.system ("mv"+ " " + source + " " + dst)

		except:
			print("failed")
			continue

