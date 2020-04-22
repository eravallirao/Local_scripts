import os
def long_text(input):
	separated_y0_from_boxes = []
	boxes_to_be_combined = []
	main_boxes = []
	Resolution_height = 450
	Deviation_Ratio = (Resolution_height/450)*6
	# Deviation_Ratio = ((Resolution_height/450) + 1)*5



	# Sorting out the obtained boxes from model
	# take second element for sort
	def takeSecond(elem):
		return elem[1]

	Input = [(178,91,559,91,559,114,178,114), (51,262,152,262,152,279,51,279), (90,236,504,235,504,255,91,256), (49,238,460,239,460,260,49,259), (49,241,382,242,382,261,49,261), (119,368,453,369,453,404,118,403), (197,362,517,362,517,396,197,395), (265,358,520,357,520,392,265,392), (119,480,453,369,453,404,118,403), (197,482,517,362,517,396,197,395), (265,485,520,357,520,392,265,392)]
	# sort list with key
	Input.sort(key=takeSecond)
	print('Sorted list:', Input)
	for i in Input:
		# print(i)
		separated_y0_from_boxes.append(i[1])

	# print(separated_y0_from_boxes)


	# Separating the boxes according to the logic of difference in y0 values
	l = 0
	j = 0

	for i in separated_y0_from_boxes:
		# print(l)
		if l == 1:
			if (i - j) <= int(Deviation_Ratio):
				boxes_to_be_combined.append(j)
			else:
				main_boxes.append(j)
			j = i
			l = l + 1
		else:
			if (i - j) <= int(Deviation_Ratio):
				boxes_to_be_combined.append(j)
				boxes_to_be_combined.append(i)
			else:
				main_boxes.append(i)

			j = i
			l = l + 1



	# cleaning the main boxes
	for i in boxes_to_be_combined:
		if i in main_boxes:
			main_boxes.remove(i)
	# print(main_boxes)



	# cleaning the combined boxes
	for i in boxes_to_be_combined:
		if i == j:
			boxes_to_be_combined.remove(i)
		j = i


	for i in main_boxes:
		if i == j:
			main_boxes.remove(i)
		j = i
	print(boxes_to_be_combined)
	print(main_boxes)


	x_boxes_to_be_combined = []

	x_main_boxes = []
	l = 0
	j = 0
	x0_values = []
	y0_values = []
	x2_values = []
	y2_values = []
	boxes_to_be_cropped = []

	index = 0

	for i in boxes_to_be_combined:
		index = index + 1

		deviationFlag = True
		if l == 0:
			j = i
			l = l + 1
			continue
		else:
			if (i - j) <= int(Deviation_Ratio):
				x_boxes_to_be_combined.append(j)
				x_boxes_to_be_combined.append(i)
				deviationFlag = False
				# print(x_boxes_to_be_combined)
			if (deviationFlag or (len(boxes_to_be_combined) == index)) :
				for d in x_boxes_to_be_combined:
					for e in Input:
						if d == e[1]:
							# print(e)
							x0_values.append(e[0])
							y0_values.append(e[1])
							x2_values.append(e[4])
							y2_values.append(e[5])
				x_main_boxes.append(i)
				x_boxes_to_be_combined = []
				final_box = [min(x0_values), min(y0_values), max(x2_values), max(y2_values)]
				# print(final_box)
				boxes_to_be_cropped.append(final_box)
				# print(boxes_to_be_cropped)
				x0_values = []
				y0_values = []
				x2_values = []
				y2_values = []
				final_box = []
			j = i
			l = l + 1



	for p in main_boxes:
		for s in Input:
			if p == s[1]:
				# print(p)
				x0_values.append(s[0])
				y0_values.append(s[1])
				x2_values.append(s[4])
				y2_values.append(s[5])
		x_boxes_to_be_combined = []
		final_box = [min(x0_values), min(y0_values), max(x2_values), max(y2_values)]
		# print(final_box)
		boxes_to_be_cropped.append(final_box)
		# print(boxes_to_be_cropped)
		x0_values = []
		y0_values = []
		x2_values = []
		y2_values = []
		final_box = []

	print(boxes_to_be_cropped)
	return boxes_to_be_cropped

	# print(x_boxes_to_be_combined)
	# print(x_main_boxes)





