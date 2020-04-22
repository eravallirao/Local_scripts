import csv
real = 0
screen = 0
printed = 0
with open('values.csv', newline='') as f:
	reader = csv.reader(f)
	for row in reader:
		print(row)
		try:
			separated_frames = str(row).split(", ")
			for frame in separated_frames:
				# print(frame)
				clean_frame = frame.replace('[', '')
				clean_frame_2 = clean_frame.replace(']', '')
				clean_frame_2 = clean_frame_2.replace("'", "")
				# print(clean_frame_2)
				clean_frame_3 = clean_frame_2.split(',')
				# print(clean_frame_3)
				# print(type(clean_frame_2))
				even_i = []
				odd_i = []
				for label in range(len(clean_frame_3)):
					# print(clean_frame_2[label])
					if label % 2:
						# print(clean_frame_2[label])
						even_i.append(clean_frame_3[label])
					else:
						# print(clean_frame_2[label]) 
						odd_i.append(clean_frame_3[label])
						if str(clean_frame_3[label]) == "0":
							# print("printing real")
							# print(clean_frame_3[label])
							real = float(real) + float(clean_frame_3[label + 1])
							# print(real)
						elif str(clean_frame_3[label]) == "1":
							# print("printing screen")
							# print(clean_frame_3[label])
							screen = float(screen) + float(clean_frame_3[label + 1])
							# print(screen)
						else:
							# print("printing printed")
							# print(clean_frame_3[label])
							printed = float(printed) + float(clean_frame_3[label + 1])
							# print(printed)
				# print(real, screen, printed)
				# print(even_i)
				# print(odd_i)
			print(real, screen, printed)
			with open('final_labesl.csv', 'a', newline='') as csvfile:
				fieldnames = ['real_value', 'screen_value', 'printed_value']
				writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
				writer.writerow({'real_value':real, 'screen_value': screen, 'printed_value':printed})
			real = 0
			screen = 0
			printed = 0
		except:
			print("something")
			with open('final_labesl.csv', 'a', newline='') as csvfile:
				fieldnames = ['real_value', 'screen_value', 'printed_value']
				writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
				writer.writerow({'real_value':"null", 'screen_value': "null", 'printed_value':"null"})
				continue

