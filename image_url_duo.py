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
		# https://vishwam.vishwamcorp.com/v1/traces/c5779255638c12ac0bb98912f3e8a606/image2.png
		# full_url = "http://oregon.vishwamcorp.com/v2/traces/"+str(xcall)+"/image1.png"
		full_url = "https://apis-az-dev.vishwamcorp.com/v1/traces/" + str(xcall)+"/image.png"
		# full_url = str(xcall)
		# full_url = "http://staging.vishwamcorp.com/v1/traces/"+str(xcall)+"/image2.png"
		# crop_url = "http://oregon.vishwamcorp.com/v2/traces/"+str(xcall)+"/image2.png"
		# polo_1 = xcall.split("/")[-1]
		# polo_2 = xcall.split("/")[-2]
		# polo_a = str(polo_2) + "_" + str(polo_1)
		fullimage_filepath = path+'/'+str(xcall) + "_" + "image.png"
		# crop_filepath = ""+str(xcall)+"_crop.png"
		with open(fullimage_filepath, 'wb') as handle1:
			response = requests.get(full_url, stream=True)
			print(response)
			print(fullimage_filepath)
			i = i + 1
			if not response.ok:
				print(response)
			for block in response.iter_content(7000):
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
