from PIL import Image
import numpy
import csv
import os
im = Image.open("0b3403d716f54f12b05b6d353948e464.png")
np_im = numpy.array(im)
print (np_im)

with open("lll.csv", 'a') as csvfile:
	fieldnames = ['x']
	# writer.writeheader()
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writerow({'x': np_im})