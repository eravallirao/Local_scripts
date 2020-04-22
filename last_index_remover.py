import os
import sys
import fileinput
import argparse


# print ("Text to search for:")
# textToSearch = input( "> " )

# print ("Text to replace it with:")
# textToReplace = input( "> " )

ap=argparse.ArgumentParser()
ap.add_argument('-i','--folder',type=str,default='',
	help="path")
args=vars(ap.parse_args())


img_path=args["folder"]
imgs_path=os.listdir(img_path)
# for images in imgs_path:

# print ("File to perform Search-Replace on:")

fileToSearch = imgs_path

for text_file in fileToSearch:
	print(str(text_file))
	text_file1 =os.path.join(img_path, text_file)
	tempFile = open( text_file1, 'r+' )
	for line in fileinput.input(text_file1):
		# try:
		line1 = line
		change = line.split(",")
		# print(change)
		change = change[:8]
		change[7] = "0"
		# baby = String.Join(",", change)
		baby = ",".join(change)
		print(baby)
		# change  = "0"
		# print(change)
		# print(str(change))
		tempFile.write(line.replace(line1, baby))
		# except:
		# 	continue
	tempFile.close()


# input( '\n\n Press Enter to exit...' )