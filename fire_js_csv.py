import requests
import json
import csv
import re

deepa = []
vinnu_final = ""
with open('sampl.json') as ff:
	resp = json.load(ff)
	# print(resp)
	resp = resp.items()
	for key, value in resp:
		# print(key)
		for key1, value1 in value.items():
			# print(key1)
			okay = ""
			for key2, value2 in value1.items():
				print(type(value2))
					# c = "["
				if key2 == "OBD_data":
					print("nothing")
				else:
					print(value2)
					okay = str(okay) + str(value2) + "@#$"

			with open('N.csv', 'a', newline='') as csvfile:
				fieldnames = ['Subject', 'XcallID','result']
				writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
				writer.writerow({'Subject': key, 'XcallID': key1,'result':okay})
				
				# for key3, value3 in value2.items():
				# 	print(type(value3))
				# 	# c = "["
				# 	if key3 == "OBD_data":
				# 		print("nothing")
				# 	else:
				# 		print(value3)
				# 		okay = str(okay) + str(value3) + "@#$"
				
					# for i in value3:
					# 	print(i)


				# v1 = list(value2)

				# deepa = str(value2)
				# deepa = list(deepa.split(","))
				# # print(deepa)
				# # print(deepa)
				# # value2 = str(value2)
				# # array = re.findall(r'[0-9]+.', value2)
				# # # deepa = value2
				# # # deepa = deepa.split("]]")
				# # print(array)
				# vinnu_final = ""
				# z = 0
				# for i in deepa:
				# 	# print (i)
					
				# 	# if i == "1" or i == "2" or i == "3" or i == "4" or i == "5" or i == "6" or i == "7" or i == "8" or i == "9" or i == "0" or i == "." in i:
				# 	if "1" or "2" or "3" or "4" or "5" or "6" or "7" or "8" or "9" or "0" or "." in i:
				# 		# print (i)
				# 		z = z + 1
				# 		if "[" in i:
				# 			k = i.replace('[', '')
				# 			vinnu = str(k) + " "
				# 			# print(k)
				# 		else:
				# 			l = i.replace(']', '')
				# 			vinnu = str(l) + " "
				# 			# print(l)
				# 		# array = re.findall(r'[0-9]+.', i)
				# 		# print(array)
				# 		# vinnu = str(i) + " "
				# 		if z % 4 == 0:
				# 			vinnu_final += (vinnu + str(","))
				# 		else:
				# 			vinnu_final += vinnu
				# 		# deepa2 = deepa2.append(i)
				# print(vinnu_final)
				with open('Nagamani.csv', 'a', newline='') as csvfile:
					fieldnames = ['Subject', 'XcallID','result']
					writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
					writer.writerow({'Subject': key, 'XcallID': key1,'result':vinnu_final})
			# 	for m in resp[item][i][k]:
			# 		print(m)
	# for s in resp:
	# 	print(s)
		# for 
		# username=s
		# writer.writerow({'Subject':username})
		# for k in resp[s]:
		# 	for l in k:
		# 		print(l)
		# 		# print(resp[s][k][l])
		# 		for m in resp[s][k]:
		# 			print(m)
					# if(m=="Rx,Ry"):
					# 	# global RxRy
					# 	RxRy=resp[s][k][l][m]
					# 	# writer.writerow({'RxRy': RxRy})
					# 	print(RxRy)
					# elif(m=="XcallID"):
					# 	# global XcallID
					# 	XcallID=resp[s][k][l][m]
					# 	# writer.writerow({'XcallID': XcallID})
					# 	print(XcallID)
					# else:
					# 	signature=resp[s][k][l][m]
						# writer.writerow({'signature':signature})
					
					


	
  
    		