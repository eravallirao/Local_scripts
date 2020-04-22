import cv2
import numpy as np


def threshold(image):
    image = cv2.threshold(image,128,255,cv2.THRESH_OTSU)[1]
    return image


def illumination_correction(image):
    _structure = cv2.getStructuringElement(cv2.MORPH_RECT,(115,155))
    image = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,_structure)
    image = cv2.bitwise_not(image)
    return image

def remove_background(image):
    image = illumination_correction(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = threshold(image)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    return image

if __name__ == "__main__":
    image = cv2.imread('1_0.jpg')
    image = remove_background(image)
    cv2.imwrite('out.jpg', image)
