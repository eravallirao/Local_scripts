# from silx.io.convert import convert

# convert("blky.h5", "shape_predictor_68_face_landmarks.dat")

import json
import csv


# with open("puli.csv", 'a', newline='') as csvfile:
#     fieldnames = ['A', 'B', 'C']
    # writer.writeheader()
    # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)


with open('all.json') as json_file:  
	data = json.load(json_file)
    # for p in data:
    #     print (p[0])
	for key, value in data.items():
		nam = key
		if "Auth" in value:
    	    # print (value["Auth"])
			joker = value["Auth"]
    	    # print ((joker))
			for key1, value1 in joker.items():
				ratios = value1["Rx,Ry"]   	    	
				sign = value1["Signature"]
				xcallid = value1["XcallID"]
				with open("puli.csv", 'a', newline='') as csvfile:
					fieldnames = ['Z', 'A', 'B', 'C']
					writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
					writer.writerow({'Z': nam, 'A': ratios, 'B': sign, 'C': xcallid})
		else:
			print("zeroooooo")
    	# for key1, values1 in key.items():
     #        print(key1)