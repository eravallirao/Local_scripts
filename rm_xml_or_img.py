import os
import io
import glob
import cv2
import csv

f_imglist = open('prod_300.txt','a')
#f_test = open('data/test_text_iraq.txt','w')

i = 0

def findjpgfile(root):
    global i
    for filename in os.listdir(root):
        full_path = os.path.join(root, filename)
        # print(full_path)
        if os.path.isfile(full_path):
            # print("entered")
            if full_path.split('.')[-1] == 'xml':
            # if full_path.split('.')[-1] == 'jpg' or full_path.split('.')[-1] == 'JPG' or full_path.split('.')[-1] == 'png':# or full_path.split('.')[-1] == 'JPG':
                # print("twice")
                full_path_txt_1 = full_path.replace('xml', 'jpg')
                full_path_txt_2 = full_path.replace('xml', 'JPG')
                full_path_txt_3 = full_path.replace('xml', 'png')
                if os.path.isfile(full_path_txt_1) or os.path.isfile(full_path_txt_2) or os.path.isfile(full_path_txt_3):
                    print("exists")
                else:
                    print(full_path)
                    continue
                # full_path_txt = full_path.replace('xml', 'JPG')
                # if os.path.isfile(full_path_txt):
                #     print("exists")
                # else:
                #     print(full_path)
                #     continue
                # full_path_txt = full_path.replace('xml', 'png')
                # if os.path.isfile(full_path_txt):
                #     print("exists")
                # else:
                #     print(full_path)
                    continue
        #         img = cv2.imread(full_path)
        #         w = img.shape[1]
        #         print(w)
        #         h = img.shape[0]
        #         print(h)
        #         with open(full_path_txt, 'r') as f:
        #             reader = csv.reader(f)
        #             j=0
        #             for line in reader:
        #                 j += 1
        #                 seperator = ','
        #                 label = seperator.join(line[8:])
        #                 # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
        #                 line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
                        
        #                 x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[:8]))
        #                 #ww = x3- x1
        #                 #hh = y3- y1
        #                 #print('{},{},{},{}'.format(x1,x3, ww, hh))
        #                 img_new = img[y1:y3, x1:x3]
        #                 print(img_new)
        #                 cv2.imwrite(str(filename) + "_" + 'front2-{}-{}.jpg'.format(i, j), img_new)
        #                 if len(label)>0:
        #                     print(label)
        #                     print(i)
        #                     print(j)
        #                     # f_imglist.write('front2-{}-{}.jpg' + ' ' + '{}\n'.format(i, j, label))
        #                     f_imglist.write(str(filename) + "_" + 'front2-' + str(i) + "-" + str(j) + '.jpg' + '_' + str(label) + '\n')

        #     i += 1
        # else:
        #     findjpgfile(full_path)

path = 'prod_300'
findjpgfile(path)
f_imglist.close()
#f_train.close()
#f_test.close()



