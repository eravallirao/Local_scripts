import cv2 as cv
import glob

imagepath = 'aadhardata/train'
imgs_names = glob.glob(imagepath+'/*.JPG')
for imgname in imgs_names:
	img = cv.imread(imgname)
	# print(imgname)
	if img is None:
		shutil.move(str(imagepath) + str(imgname), "/Users/sukshi/Downloads/Z_gold_poi/" + str(ll))
		print("weghjsadnmzqjaksznijakNS,ZMAJKzN<XM akjZXNm, askzJNXm asZJXnm")
		print(imgname)