import shutil
import os

source = '/Users/sukshi/Downloads/nn'
dest1 = '/Users/sukshi/Downloads/nnn/'

j = 0
files = os.listdir(source)

for f in files:
	for i in os.listdir(source+'/'+f):
		j = j + 1
		# print(source+'/'+f+'/'+i)
		# print(f)
		# print(i)
		i = i.split(".")[0] + "_" + str(j) + ".jpg"
		# try:
		shutil.move(source+'/'+f+'/'+i, dest1)
		# except:
		print("check")
		continue
    