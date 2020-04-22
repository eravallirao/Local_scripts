import os
from datetime import datetime
from time import time
import argparse
import keras
import keras.backend as K
from keras.models import load_model, model_from_json
import detect_face
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
import utils
from keras.preprocessing import image as image_loader
from keras.models import Model
# from face_model import FaceModel



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def draw_key_points(image_path, image_orig, bounding_box, keypoints):
    image = image_orig.copy()
    cv2.rectangle(image, (int(bounding_box[0]), int(bounding_box[1])), (int(bounding_box[2]), int(bounding_box[3])), (0, 255, 0),3)
    for i in range(5):
        cv2.circle(image, (keypoints[i], keypoints[i + 5]), 5, (0, 40, 255), 3)
    cv2.imwrite(image_path + "_key_points.jpg", image)

def get_faces(folder):
    dirs = os.listdir(folder)
    all_dirs = [os.path.join(folder,x) for x in dirs]
    faces = []
    for f in all_dirs:
        faces.append(f)    
    return faces



global fnet, graph, pnet, rnet, onet, minsize, threshold, factor,resnet_model

__sess = K.get_session()
pnet, rnet, onet = detect_face.create_mtcnn(__sess, './trained/mtcnn/')
print(type(pnet))
minsize = 50 # minimum size of face
threshold = [ 0.5, 0.6, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
graph = K.get_session().graph
def crop_face(image_path):
    if not os.path.isfile(image_path):
        return False , "no file found"
    image = cv2.imread(image_path)

    height, width, channels = image.shape

    height_up = 0.07*height
    height_down = 0.11*height
    width_side = 0.06*width
    _max = 2000
    if height > _max or width > _max:
        scaling_factor = _max / float(height)
        if _max/float(width) < scaling_factor:
            scaling_factor = _max / float(width)
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    image_gray = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image_gray,cv2.COLOR_GRAY2BGR)


    t1_d = time()
    bounding_boxes, points = detect_face.detect_face(image_gray, minsize, pnet, rnet, onet, threshold, factor)

    t2_d = time()
    total_time_d = t2_d - t1_d
    print(bounding_boxes)
    
    if len(bounding_boxes) > 0:
        bounding_box = bounding_boxes[0]
        keypoints = points
        top_h_point = int(keypoints[5])

        # print("bounding box left top co-ordinates : ",int(bounding_box[0]), int(bounding_box[1]))
        # print("bounding box right bottom co-ordinates : ",int(bounding_box[2]), int(bounding_box[3]))
        # print("left eye xy co-ordinaates : ", int(keypoints[0]),int(keypoints[5]))
        # print("right eye xy co-ordinates : ",int(keypoints[1]),int(keypoints[6]))
        # print("nose xy co-ordinates :",int(keypoints[2]),int(keypoints[7]))
        # print("mouth left xy co-ordinates : ",int(keypoints[3]),int(keypoints[8]))
        # print("mouth right xy co-ordinates : ",int(keypoints[4]),int(keypoints[9]))
        #draw_key_points(image_path,image,bounding_box,keypoints)
        # h2 = int(bounding_box[1])
        # h1 = int(bounding_box[3])
        # w2 = int(bounding_box[0])
        # w1 = int(bounding_box[2])
        # print(height,width)
        # print(height_up, height_down, width_side)
        # print("4673876328wi7382whdjsxniudwskajdnxisakjzndxm wijsaknzmxwdjsaknmzxwdnsjkzcmxwdnsjkcmxednwsjkzx", h2,h1,w2,w1)
        img_cropped = image[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]
        #img_cropped = image[int(h2):int(h1), int(w2):int(w1)]
        h, w = image.shape[:2]
        h0 = int(0.15*(h))
        h1 = h
        w0 = 0
        w1 = w
        # fin_img = img_cropped[h0:h1, w0:w1]
        fin_img = image[top_h_point:h1, w0:w1]
        cv_interpolation = cv2.INTER_LANCZOS4
        cropped = cv2.resize(fin_img, dsize=(224, 224), interpolation=cv_interpolation)
        # cropped = image[top_h_point:h1, w0:w1]

        #if h < 10 or w < 10:
        #    return False , "no face found", None, None
        #resized_image = cv2.resize(img_cropped, (168, 224))
        #image_padded = cv2.copyMakeBorder( resized_image, 0, 0, 28, 28, cv2.BORDER_CONSTANT)
        #print("------------------------------*********************",image_path)
        fl2 = image_path.split("/")[-1]
        #cv_interpolation = cv2.INTER_LANCZOS4
        fl2 = fl2.split(".")[0]
        #cropped = cv2.resize(img_cropped, dsize=(224, 224), interpolation=cv_inte#rpolation)
        cv2.imwrite("/home/sukshi/Desktop/desk/beardmodel/beardcrop/" + str(fl2) + ".jpg",cropped)
    if len(bounding_boxes) == 0:
        print("no face found")


# def compare(folder):

#     face_pairs = get_faces(folder)
#     print('total faces found',len(face_pairs))
#     for _imgs in face_pairs:
#         print("---")
#         print("image : ",_imgs)
#         try:
#             face1 = crop_face(_imgs)
#         except:
#             print("shifted")
#             continue       
def compare(folder):

    face_pairs = get_faces(folder)
    print('total faces found',len(face_pairs))
    for _imgs in face_pairs:
        print("---")
        print("image : ",_imgs)
        face1 = crop_face(_imgs)
 


def main(args):
    compare(args.folder)
print("something")
#try:
 #   compare(args.folder)
#except:
 #   print("hfirfn") 
 #   continue
  #  print("done")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Comparison eval')
    parser.add_argument('--folder', type=str, required=True,
                    help='Folder with subfolders of face pair')
    args = parser.parse_args()
    main(args)