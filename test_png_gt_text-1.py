from os.path import exists, join, isdir
from os import remove, listdir
from PIL import Image
import glob
import argparse
import csv
import pytesseract
import cv2



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--file", type=str, default='../data/all.csv',
                help="path to ground truth text file")
ap.add_argument("-d", "--directory", type=str, default="../data/all",
                help="path to image directory")

args = vars(ap.parse_args())

gt_path = args['file']
img_dir = args['directory']

def fix_number(str):
    res =''
    for i in range(0, len(str)):
        if str[i] == '8':
            res += 'B'
        elif str[i] == '1':
            res += 'l'
        elif str[i] == '0':
            res += 'O'
    return res

def fix_alpha(str):
    res = ''
    for i in range(0, len(str)):
        if str[i] == 'B':
            res += '8'
        elif str[i] == 'O':
            res += '0'
        elif str[i] == 'l':
            res += '1'
        else:
            res += str[i]
    return res

def correct_string(str):
    segs = str.split(' ')
    res=''
    i=0
    for seg in segs:
        numbers = sum(c.isdigit() for c in seg)
        alphas = sum(c.isalpha() for c in seg)
        if i==0 and numbers ==0 and alphas == 0 and len(seg) ==1 :
            i=1
            continue
        if numbers == 1 and alphas > 2:
            seg = fix_number(seg)
        elif alphas == 1 and numbers > 2:
            seg = fix_alpha(seg)
        res = res + seg + ' '
        i +=1
    return res.rstrip()

### CxImage Adaptive thresholding implement
### 2020-03-28 LTI
def Adaptive_threshold(img, nBoxSize):
    h, w, c = img.shape;
    mh = (int)((h+ nBoxSize -1 )/nBoxSize)
    mw = (int)((w+ nBoxSize -1 )/nBoxSize)
    fGlobalLocalBalance = 0.5


def main():
    if not exists(gt_path):
        print("No such file: {}".format(gt_path))
        return
    if not isdir(img_dir):
        print("No such folder: {}".format(img_dir))
        return

    # read gt file and generate each gt file for training
    file_counts = len(list(glob.glob("{}/*.jpg".format(img_dir))))
    right_cnt = 0
    custom_oem_psm_config = r'-c tessedit_char_blacklist="_" --oem 3 --psm 6 --tessdata-dir "/mnt/hgfs/win_share/my_tesseract/mybest"'

    with open(gt_path, 'rt', encoding='utf-8') as pf:
        with open('out.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            csv_reader = csv.reader( pf, delimiter = ",")
            i=0
            for line in csv_reader:  # <image file name>.jpg,<text>
                #terms = line.strip().split(',')
                #if len(terms) != 3:
                #    continue
                file_name = line[0]
                text = line[1]
                if text == '':
                    writer.writerow(['file_name', 'text', 'res', ''])
                    continue

                file_path = join(img_dir, file_name)
                if not exists(file_path):
                    continue

                i+=1
                print("\33[2K\r[+] Processing [{}/{}] --- {}".format(i, file_counts, file_name), end='')

                # save line gt file
                print("\nG T:{}".format(text))
                with open(join(img_dir, file_name[:-4]+'.txt'), 'w', newline='') as txt_f:
                    txt_f.write(text)


                try:
                    im = Image.open(file_path)
                    res = pytesseract.image_to_string(im, lang='eng', config=custom_oem_psm_config).strip()

                    res = correct_string(res)

                    print("OCR:{}".format(res))
                    if text == res:
                        right_cnt +=1
                    else:
                        im.save(join('./error', res +'.jpg'))

                    writer.writerow([file_name, text, res, text == res.strip()])
                except:
                    pass
    if right_cnt > 0:
        print("acc= %f", right_cnt / i)

    print("\nProcessing has finished successfully!")



if __name__ == '__main__':
    main()
