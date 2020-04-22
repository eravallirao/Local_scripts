import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import csv
import time


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result



MODEL_NAME = 'Aadhar_photo_detection.pb'
IMAGE_NAME = 'aa'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(MODEL_NAME, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
for fl in os.listdir(IMAGE_NAME):
    if fl == ".DS_Store" or fl == "_DS_Store":
        print ("sorry")
    else:
        images2 = os.path.join(IMAGE_NAME,fl)
        image = cv2.imread(images2)
        shape=image.shape
        height = shape[0]
        width = shape[1]
        height_increase = int((height/666)*35)
        width_increase_right = int((width/967)*35)
        width_increase_left = 0
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
        box_2=np.squeeze(boxes)
        result =np.squeeze(scores)
        print(box_2.shape)
        for i in range(box_2.shape[0]):
            if result[i]>0.5:
                print(box_2[i])
                (xmin, xmax, ymin, ymax) = (box_2[i][1] * shape[1], box_2[i][3] * shape[1],
                    box_2[i][0] * shape[0], box_2[i][2] * shape[0])
                print("xmin = ", xmin)
                print("xmax= ", xmax)
                print("ymin= ", ymin)
                print("ymax= ", ymax)
                cropped_img=image[int(ymin):int(ymax),int(xmin):int(xmax)]
                ymin = ymin - height_increase
                ymax = ymax + height_increase
                # xmin = width - xmin
                xmax = xmax + width_increase_right
                color = (255,255,255)
                img_with_border = cv2.copyMakeBorder(cropped_img, int(height_increase), int(height_increase), int(xmin), int(width_increase_right), cv2.BORDER_CONSTANT, value=color)
                cv2.imwrite(str(fl),img_with_border)
                with open('1.csv', 'a', newline='') as csvfile:
                  fieldnames = ['imagename','xmin','xmax','ymin','ymax']
                  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                  writer.writerow({'imagename': fl,'xmin':xmin,'xmax':xmax,'ymin':ymin, 'ymax':ymin})