import os
import argparse
import dlib
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
ap.add_argument("-c", "--csvc", type=str, default="",
	help="path to input image file")

args = vars(ap.parse_args())


images = args["image"]
csvs = args["csvc"]
print(images)
print(csvs)


# for fl in os.listdir(images):
# 	#print(fl)
# 	if fl == ".DS_Store" or fl == "_DS_Store":
# 		#print(fl)
# 		print("stupid files")

# 	else:
# 		images2 = os.path.join(images,fl)
# 		print(fl)	
# 		frame = cv2.imread(images2)
# 		# detector = dlib.get_frontal_face_detector()
# 		# try:			
# 		fl2 = fl.split(".")[0]
# 		# cv2.imwrite( "cropped" + ".jpg",face)
# 		cv_interpolation = cv2.INTER_LANCZOS4
# 		# cropped = cv2.resize(frame,(224,224))
# 		cropped = cv2.resize(frame, dsize=(300, 300), interpolation=cv_interpolation)
# 		# cv2.imwrite( str(fl) + ".jpg",cropped)
# 		cv2.imwrite( str(images) + "/" + str(fl2) + ".jpg",cropped)






# # importing the requests library 
# import requests 
  
# # defining the api-endpoint  
# API_ENDPOINT = "http://oregon.vishwamcorp.com/v1/check_qr_code"
  
# # your API key here 
# API_KEY = 'dkyc'
  
# # data to be sent to api 
# data = {'app_id':API_KEY} 

# multiple_files = { 'image': open('image1.png','rb'),

# 'image2': open('image1.png','rb') }
# # r = requests.post(url, files=multiple_files)
# # r.text
  
# # sending post request and saving response as response object 
# r = requests.post(url = API_ENDPOINT, data = data, files=multiple_files) 
  
# # extracting response text  
# pastebin_url = r.text

# print("response code:%s",r.status_code)
# print("The pastebin URL is:%s"%pastebin_url) 