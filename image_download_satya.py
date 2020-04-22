import os
import requests
import cv2
import argparse


# print(os.getcwd())

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type=str, default="",
	help="path to input txt file")
args = vars(ap.parse_args())


path=os.getcwd()

txtpath= args["file"]

print(txtpath)
with open(txtpath) as fp:
	lines = fp.read().split("\n")
	i = 0
	for xcall in lines:
		print(xcall)
		# full_url = xcall
		# xcallid = xcall.split("/")[-2]
		xcallid = xcall
		# https://apis-az-dev.vishwamcorp.com/v1/traces/",A1726,"/image.png
		full_url = "https://apis-az-preprod.vishwamcorp.com/v1/traces/"+str(xcall)+"/image.png"
		# full_url = "https://apis-az-dev.vishwamcorp.com/v1/traces/" +str(xcall)+"/image.png"
		# full_url = "http://Vishwam.vishwamcorp.com/v1/traces/"+str(xcall)+"/image.png"
		# crop_url = "http://oregon.vishwamcorp.com/v2/traces/"+str(xcall)+"/image2.png"
		fullimage_filepath = path+'/'+str(xcallid)+".png"
		# crop_filepath = ""+str(xcall)+"_crop.png"
		with open(fullimage_filepath, 'wb') as handle1:
			response = requests.get(full_url, stream=True)
			print(response)
			print(fullimage_filepath)
			i = i + 1
			if not response.ok:
				print(response)
			for block in response.iter_content(1024):
				if not block:
					break
				handle1.write(block)
		# with open(crop_filepath, 'wb') as handle2:
		# 	response = requests.get(crop_url, stream=True)
		# 	print(response)
		# 	print(crop_filepath)
		# 	i = i + 1
		# 	if not response.ok:
		# 		print(response)
		# 	for block in response.iter_content(1024):
		# 		if not block:
		# 			break
		# 		handle2.write(block)
