from PIL import Image
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())


images = args["image"]	

# cv2.imshow('Scaling - Linear Interpolation', img_scaled) img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)


for fl in os.listdir(images):
	#print(fl)
	if fl == ".DS_Store" or fl == "_DS_Store":
		#print(fl)
		print("stupid files")

	else:
		images2 = os.path.join(images,fl)
		column = Image.open(images2)
		gray = column.convert('L')
		blackwhite = gray.point(lambda x: 0 if x < 140 else 255, '1')
		fl2 = fl.split(".")[0]
		blackwhite.save(str(fl2) + ".jpg")