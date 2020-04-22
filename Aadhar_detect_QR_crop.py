import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import csv
import time
MODEL_NAME = 'frozen_inference_graph4.pb'
# IMAGE_NAME = 'tt'
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

try:
    image = cv2.imread(Filename)
    shape=image.shape
    e_height = shape[0]
    e_width = shape[1]
    print(e_height, e_width)
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

            height = ymax - ymin
            width = xmax - xmin
            width_2 = xmax
            height_2 = int(ymax*(2.9/3))


            width_0 = int((width)*(1.8/3)) + xmin
            height_0 = int((height)*(1.1/3)) + ymin
            # width_1 = int(width)
            # height_1 = int((height)*(2.9/3))



            cropped_img=image[int(height_0):int(height_2),int(width_0):int(width_2)]
            img_scaled = cv2.resize(cropped_img,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(fl,img_scaled)
except:
    print("corrupted image")
    continue