import requests
import json
import csv
import re
import shutil
import os


source = '/Users/sukshi/Downloads/ocr_images'
dest1 = '/Users/sukshi/Downloads/below_900/'


with open('POA_below_900.txt') as f:
	lines = f.readlines()
	for j in lines:
		j = j.split(".")[0]
		j = j + ".png"
		shutil.move(source+'/'+ j, dest1)
		print(j)