import numpy as np
from PIL import Image
from ISR.models import RDN
import os
import argparse
# import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())

images = args["image"]
# image_path_output = '/Users/sukshi/Downloads/t_r'

rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')
for fl in os.listdir(images):
	#print(fl)
	if fl == ".DS_Store" or fl == "_DS_Store":
		#print(fl)
		print("stupid files")
	else:
		try:
			images2 = os.path.join(images,fl)
			img = Image.open(str(images2))
			lr_img = np.array(img)
			fl2 = fl.split(".")[0]		
			sr_img = rdn.predict(lr_img)
			ll = Image.fromarray(sr_img)
			ll.save(str(images) + '/' + str(fl2) + '.jpg')
			print("saved")
		except:
			print("skipped")
			continue




