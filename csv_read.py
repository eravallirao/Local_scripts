import csv


Chotu = []
with open('LandmarksPoints.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        # print(type(row))
        print(row[0])
        print("row")
        # for i in row:
        # 	print (i)
        # 	print("some")
        	# for z in i:
        	# 	print(z)
        # chotu = row
        # print(chotu)
        # print(row[2])

csvFile.close()