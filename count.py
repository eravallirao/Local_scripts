# count_1=0
# with open ('combined_rand.txt','rb') as f:
#     for line in f:
#         count_1+=1
# print (count_1)

# import os
# import shutil



# filepath = 'first_gold_POI.txt'
to_check = ' "#&\'()*,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]_abcdefghijklmnopqrstuvwxyz}{ʼ'
to_remove = '~^!$%+<=>?@`'
# ^!$%+<=>?@`
# for i in to_check:
# 	print (i)

import re

lines = [line.rstrip('\n') for line in open('11a_char.txt')]
# print(lines)
count = 0
filenaming = 0
for ll in lines:
	# print(ll)
	count+=1
	for i in to_remove:
	# 	# print (i)
		if i in ll:
	# 		filenaming+=1
			try:
				# line = re.sub('', '>', ll)
				print(str(ll).replace(str(i), ""))
				
			except:
				continue

		else:
			continue
	print(ll)




