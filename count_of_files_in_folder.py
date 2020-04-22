import os
import csv

images = "/Users/sukshi/Downloads/State_images/"

for sub_path in os.listdir(images):
    if sub_path == ".DS_Store" or sub_path == "_DS_Store":
        #print(fl)
        print("stupid files")
    else:
        print(sub_path)
        full_path = images + str(sub_path)
        print(full_path)
        full_path_count_path = os.listdir(full_path)
        file_count = len(full_path_count_path)
        print(file_count)
        with open("count.csv",'a',newline='') as csvfile:
            fieldnames = ['Folder_name', 'count',]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Folder_name': sub_path,'count':file_count}) 

