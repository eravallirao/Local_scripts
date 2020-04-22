import requests
import json
import csv

xcallid = ""
# r='obd.json'
# resp=r.json()
# print(resp['Arun-down']['Auth']['3n4mkutwSZ']['Rx,Ry'])
# print(type(resp))
# f=open('sheet4.csv','w')
with open('obdc.json') as ff:
	resp = json.load(ff)
	#print(resp)
	resp = resp.items()
	for key, value in resp:
		xcallid = key
		print(key)
		# print(value["1"])
		if not(value):
			print(none)
		else:
			try:
				for key1, value1 in value.items():
					if key1 in "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30":
						print("1", key1)
						keying = key1
						frame_score = value[key1]["frameScore"]
						image_link = value[key1]["imageLink"]
						with open('three.csv', 'a', newline='') as csvfile:
							fieldnames = ['xcall', 'keyi', 'frameScore', 'imageLink']
							writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
							writer.writerow({'xcall': xcallid, 'keyi':keying, 'frameScore': frame_score, 'imageLink':image_link})
					else:
						if key1 == "TotalFrames":
							total_frame = value1
						elif key1 == "clientResult":
							clien_result = value1
						elif key1 == "inputType":
							input_type = value1
						elif key1 == "phoneModel":
							phone_model = value1
						elif key1 == "printFrames":
							print_frames = value1
						elif key1 == "realFrames":
							real_frame = value1
						elif key1 == "screenFrames":
							scree_frame = value1
						elif key1 == "obdString":
							obd_string = value1
				with open('three.csv', 'a', newline='') as csvfile:
					fieldnames = ['TotalFrames', 'clientResult', 'inputType', 'phoneModel', 'printFrames', 'realFrames', 'screenFrames', 'obdString']
					writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
					writer.writerow({'TotalFrames': total_frame, 'clientResult': clien_result, 'inputType':input_type, 'phoneModel': phone_model, 'printFrames': print_frames, 'realFrames':real_frame, 'screenFrames': scree_frame, 'obdString': obd_string})
			except:
				print("continue")
		  
			    		