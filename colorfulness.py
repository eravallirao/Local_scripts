from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import csv

def image_colorfulness(image):
	# split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))

	# compute rg = R - G
	rg = np.absolute(R - G)

	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)

	# compute the mean and standard deviation of both `rg` and `yb`
	(rgMean, rgStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))

	# combine the mean and standard deviations
	stdRoot = np.sqrt((rgStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rgMean ** 2) + (ybMean ** 2))

	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
ap.add_argument("-f", "--file", type=str, default="",
	help="path to text file")
args = vars(ap.parse_args())

file = args["file"]

# initialize the results list
print("colorness of images were")
results = []

with open(file, 'w') as writeFile:
	writer = csv.writer(writeFile)
	writer.writerow(["imagename", "score"])


	# loop over the image paths
	for imagePath in paths.list_images(args["images"]):
		# load the image, resize it (to speed up computation), and
		# compute the colorfulness metric for the image
		image = cv2.imread(imagePath)
		# image = imutils.resize(image, width=250)
		C = image_colorfulness(image)

		line = [os.path.basename(imagePath), C]
		writer.writerow(line)

		# print(os.path.basename(imagePath),":",C)
		#appending the results
		results.append((os.path.basename(imagePath),C))


	
	#print(results)
# sorting the results in descendint order
results = sorted(results, key=lambda x: x[1], reverse=True)
descColor = [r for r in results[:25]]

for i in descColor[:]:
	print(i)