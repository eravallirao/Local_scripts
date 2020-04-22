import os
import io
import glob
import cv2
import csv
import argparse
import numpy as np
from PIL import Image


#f_test = open('data/test_text_iraq.txt','w')

ap=argparse.ArgumentParser()
ap.add_argument('-s','--source', type=str, default='true',
 help="path to source folder")
ap.add_argument('-d','--destination', type=str, default='true',
 help="path to destination folder")
args=vars(ap.parse_args())
src=args['source']
dst=args['destination']

os.mkdir(dst)

i = 0


def findjpgfile(src,dst):
    global i
    for filename in os.listdir(src):
        full_path = os.path.join(src, filename)
        print(full_path)
        if os.path.isfile(full_path):
            print("entered")
            if full_path.split('.')[-1] == 'jpg':# or full_path.split('.')[-1] == 'JPG':
                print("twice")
                full_path_txt = full_path.replace('jpg', 'txt')
                if os.path.isfile(full_path_txt):
                    aa=''
                else:
                    continue
                img = cv2.imread(full_path)
                
                imge_name=filename.split('.')
                a=str(imge_name[0])
                print(a)
                pil_img = Image.open(full_path)
                print(type(pil_img))
                w = img.shape[1]
                print(w)
                h = img.shape[0]
                print(h)
                with open(full_path_txt, 'r') as f:
                    reader = csv.reader(f)
                    j=0
                    for line in reader:
                        j += 1
                        seperator = ','
                        label = seperator.join(line[8:])
                        # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
                        line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
                        
                        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[:8]))
                    
                        img_new = img[y1:y3, x1:x3]
                        # print(img_new)
                        kernel_size = 10
                        kernel_h = np.zeros((kernel_size, kernel_size))
                        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
                        kernel_h /= kernel_size 
                        print(kernel_h.shape)
                        horizonal_mb = cv2.filter2D(img_new,-1, kernel_h)
                        horizonal_mb = cv2.cvtColor(horizonal_mb, cv2.COLOR_BGR2RGB)
                        new_im = Image.fromarray(horizonal_mb)

                        pil_img.paste(new_im, [x1,y1])
                        pil_img.save(dst+'/'+a+"_text_blur.jpg")

                        # cv2.imwrite('k'+str(filename) + "_" + 'front2-{}-{}.jpg'.format(i, j), horizonal_mb)
                        # if len(label)>0:
                        #     print(label)
                        #     print(i)
                        #     print(j)
                        #     # f_imglist.write('front2-{}-{}.jpg' + ' ' + '{}\n'.format(i, j, label))
                        #     f_imglist.write(str(filename) + "_" + 'front2-' + str(i) + "-" + str(j) + '.jpg' + '_' + str(label) + '\n')

            i += 1
        else:
            findjpgfile(full_path)

# path = 'sep_29_200'
findjpgfile(src,dst)
# f_imglist.close()
#f_train.close()
#f_test.close()