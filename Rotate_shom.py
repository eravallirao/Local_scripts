def crop_minAreaRect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    
    if angle < -45:
        angle = angle + 90
        size = (size[1], size[0])
    size = (size[0], size[1] * 1.28)
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop
	
def example():
    rotated_rect = cv2.minAreaRect(pts)
    rect = cv2.boundingRect(pts)
    output_roi = crop_minAreaRect(img.copy(), rotated_rect)
    output_roi = resize_for_ocr(output_roi)
    cropped_regions.append([output_roi, rect])