import os
import csv
import shutil

images = "/Users/sukshi/Downloads/State_images/"
desti = "/Users/sukshi/Downloads/new_ping/"

j = 0

for sub_path in os.listdir(images):
    j = j + 1
    new_folder_name = "frface_" + str(j)
    if sub_path == ".DS_Store" or sub_path == "_DS_Store":
        #print(fl)
        print("stupid files")
    else:
        # print(sub_path)
        full_path = images + str(sub_path)
        # print(full_path)
        i = 0
        new_full = str(desti) + str(new_folder_name)
        os.mkdir(new_full)
        for file_name in os.listdir(full_path):
            i = i + 1
            print(file_name)
            print(i)
            print(str(full_path) + "/" + str(file_name))
            source = str(full_path) + "/" + str(file_name)
            if i > 2:
                break
            else:
                shutil.copy(source, new_full)
                print("file moving")




