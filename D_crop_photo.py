import os
import random
import argparse
import cv2


lines_p = []
bunch2 = []


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--textfile", type=str, default="",
	help="path to input file")
args = vars(ap.parse_args())


files = args["textfile"]	
# lines_2 = [line.rstrip('\n') for line in open("/Users/sukshi/Downloads/D_PHO_spo/" + str(fl))]
# print(lines_2)

for fl in os.listdir(files):
	#print(fl)
	if fl == ".DS_Store" or fl == "_DS_Store":
		#print(fl)
		print("stupid files")
	else:
		lines_2 = [line.rstrip('\n') for line in open("/Users/sukshi/Downloads/D_PHO_spo/" + str(fl))]
		print(len(lines_2))
		for i in range(len(lines_2)):
			abc = "(" + str(lines_2[i]) + ")"
			bunch2.append(abc)
			print(bunch2)
		print(bunch2.sort())
		def takeSecond(elem):
			print(elem.split(",")[1])
			# bunch2.append(elem.split(",")[1])
			# return elem.split(",")[1]

		Input = bunch2
		Input2 = bunch2
		# sort list with key
		Input.sort(key=takeSecond)
		print('Sorted list:', Input)
		bunch2 = []
			
				



# random.shuffle(lines)
# for ll in lines:
# 	bunch.append(ll)
# print(bunch)

# lines = [line.rstrip('\n') for line in open('2489adf283431f183d558d43e7c05f00.txt')]


# def takeSecond(elem):
# 	print(elem.split(",")[1])
# 	bunch2.append(elem.split(",")[1])
# 	return elem.split(",")[1]

# Input = bunch
# Input2 = bunch2
# # sort list with key
# Input.sort(key=takeSecond)
# print('Sorted list:', Input)


