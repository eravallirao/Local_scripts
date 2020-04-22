# from predict import get_model_api
import time
import cv2
import numpy as np
from glob import glob
from PIL import Image
import os, sys
# import Image

# im = Image.open("image.jpg")
# x = 3
# y = 4

# pix = im.load()
# print pix[x,y]

# attention_ocr = get_model_api()

def attention_ocr(_):
    pass

def threshold(image):
    image = cv2.threshold(image,128,255,cv2.THRESH_OTSU)[1]
    cv2.imwrite('nov2_rota_c/otsus_threshold.jpg', image)
    return image


def illumination_correction(image):
    _structure = cv2.getStructuringElement(cv2.MORPH_RECT,(25,15))
    cv2.imwrite('nov2_rota_c/_structure.jpg', _structure)
    image = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,_structure)
    cv2.imwrite('nov2_rota_c/_blackhat.jpg', image)
    image = cv2.bitwise_not(image)
    cv2.imwrite('nov2_rota_c/_bitwise.jpg', image)
    _structure = cv2.getStructuringElement(cv2.MORPH_RECT,(7,1))
    cv2.imwrite('nov2_rota_c/_getstruct2.jpg', _structure)
    image = cv2.morphologyEx(image,cv2.MORPH_ERODE,_structure)
    cv2.imwrite('nov2_rota_c/illuminated.jpg', image)
    return image

def save_text(filename, content):
    with open(filename, 'w+', encoding='utf-8') as f:
        f.write(content)

def remove_starting_whitespaces(text_line):
    count = 0
    for c in text_line:
        if c == ' ':
            count += 1
        else:
            break
    text_line = text_line[count:]
    return text_line

def resize_scaled(image, height = 32):
    h, w = image.shape[:2]
    new_width = int((height * w)/h)
    image = cv2.resize(image,(new_width,height))
    
    return image

def resize_for_ocr(image, pixels_values):
    image = resize_scaled(image,26)
    # image = resize_scaled(image,32)
    # a, b, c = pixels_values
    print("printing fomr resizingdsxwdsdzxdszxx")
    # print(a,b,c)
    # image = cv2.copyMakeBorder(image,3,3,4,4,cv2.BORDER_CONSTANT,value=(255, 255, 255))
    # image = cv2.copyMakeBorder(image,4,4,3,3,cv2.BORDER_REFLECT)
    return image

def get_lines_segments(line_image):
    line_seg = []
    _line_image = line_image.copy()
    _line_image = illumination_correction(_line_image)
    _line_image = threshold(_line_image)
    _width_limit = 240
    hist = cv2.reduce(_line_image,0, cv2.REDUCE_AVG).reshape(-1)
    max_th = max(hist) * .8
    H, W = _line_image.shape[:2]
    th = max_th
    to = [y for y in range(W) if hist[y] >= th]
    _from = 0
    _to = 1

    for i,x in enumerate(to,1):
        # cv2.line(line, (x,0), (x,H),(0,0,255), 1)
        _width = x - _from
        if _width <= _width_limit:
            _to = x

        else:
            _line = line_image[:,_from:_to]
            _from = _to
            _to = x
            if _line.shape[1] > 0 and _line.shape[0] > 0:
                _line = cv2.copyMakeBorder(_line,0,0,2,2,cv2.BORDER_CONSTANT,value=255)
                line_seg.append(_line)
            
        if x == to[-1] and x != _from:
            _line = line_image[:,_from:]
            if _line.shape[1] > 0 and _line.shape[0] > 0:
                _line = cv2.copyMakeBorder(_line,0,0,2,2,cv2.BORDER_CONSTANT,value=255)
                line_seg.append(_line)

    return line_seg

def do_ocr_raw(img_for_ocr):
    img_for_ocr = resize_for_ocr(img_for_ocr)
    out_text = ''
    confidence = 0.0
    h, w = img_for_ocr.shape[:2]
    if w > 350:
        line_segments = get_lines_segments(img_for_ocr)
        _tmp_ocr_out = []
        for j, _line in enumerate(line_segments):
            _ocr_out, _confidence = attention_ocr(_line)
            _ocr_out = remove_starting_whitespaces(_ocr_out)
            _ocr_out = _ocr_out.strip()
            confidence = (confidence + _confidence) / (j+1)
            _tmp_ocr_out.append(_ocr_out)
        out_text = ' '.join(_tmp_ocr_out)
    else:
        out_text, confidence = attention_ocr(img_for_ocr)
        out_text = remove_starting_whitespaces(out_text)
        out_text = out_text.strip()

    return out_text, confidence

global counter
counter = 0

def find_largest_contour(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    return cnt

def crop_minAreaRect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    print(size)
    print(center)
    if angle < -45:
        angle = angle + 90
        size = (size[1], size[0])
    size = (size[0], size[1] * 1.28)
    center, size = tuple(map(int, center)), tuple(map(int, size))
    print("changeing sizes and center")
    print(size)
    print(center)

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height),borderValue=(255,255,255))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_rot

def auto_rotate_text_line(line_image, _file, pixels_values):
    global counter
    image = line_image.copy()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = illumination_correction(image)
    image = threshold(image)
    _structure = cv2.getStructuringElement(cv2.MORPH_RECT,(25,1))
    cv2.imwrite('nov2_rota_c/final_structue.jpg', _structure)
    image = cv2.morphologyEx(image,cv2.MORPH_ERODE,_structure)
    cv2.imwrite('nov2_rota_c/morph_erode.jpg', image)
    image = cv2.morphologyEx(image,cv2.MORPH_DILATE,_structure)
    cv2.imwrite('nov2_rota_c/morph_dilate.jpg', image)
    image = cv2.bitwise_not(image)
    cv2.imwrite('nov2_rota_c/bitwsie.jpg', image)
    pts = find_largest_contour(image)
    rotated_rect = cv2.minAreaRect(pts)
    #cv2.imwrite('nov2_rota_c/rotated_rect.jpg', rotated_rect)
    rect = cv2.boundingRect(pts)
    cv2.imwrite('nov2_rota_c/rect.jpg', rect)
    output_roi = crop_minAreaRect(line_image.copy(), rotated_rect)
    output_roi = resize_for_ocr(output_roi, pixels_values)

    cv2.drawContours(line_image,[pts],-1,(147,20,255),1)
    # cv2.imwrite('uu/{}.jpg'.format(counter), line_image)
    counter += 1
    print(_file)
    fl2 = str(_file).split("/")[-1]
    fl2 = fl2.split(".")[0]
    # cv2.imwrite('nov2_rota/' + str(_file) + "_" + '{}.jpg'.format(counter), output_roi)
    cv2.imwrite('nov2_rota/pass/' + str(fl2) + "_" + '.jpg', output_roi)
    counter += 1
    # to replace the original file with cropped and angle corrected line
    # cv2.imwrite(_file, output_roi)



def main(folder):
    files = glob('{}/*.jpg'.format(folder))
    print(files)
    total_time = 0
    for _file in files:
        image = cv2.imread(_file)
        im = Image.open(_file)
        pix = im.load()
        pixels_values = pix[2,2]
        print("printing pixel values")
        print(pix[2,2])
        print("printed pixel values")
        # ocr_out, confidence = do_ocr_raw(image)
        # print(ocr_out,confidence)
        t1 = time.time()
        if image is not None:
            auto_rotate_text_line(image, _file, pixels_values)
        t2 = time.time() - t1
        total_time += t2

    # print('total images {}, time taken {}'.format(len(files),total_time))
    

if __name__ == '__main__':
    main('fg/pass')
