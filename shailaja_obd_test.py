# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import os
import csv
import sys
import time



def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=224,
                                input_width=224,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()

  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  
  input_height = 224
  input_width = 224
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"


  


  file_name = sys.argv[1]  # prints python_script.py
  csvs = sys.argv[2]  # prints var1
  model_file = sys.argv[3]
  # seperation = sys.argv[4]

  print (file_name)
  print(csvs)
  print(model_file) 



  # model_file = "test6.pb"
  label_file = "output_labels.txt"


  graph = load_graph(model_file)

  # images = args["image"]
  start = time.time()
with open(str(csvs), 'a', newline='') as csvfile:
  fieldnames = ['imagename', 'Person','Screen','Print']

  with tf.Session(graph=graph) as sess:
    for fl in os.listdir(file_name):
      # try:
      if fl == ".DS_Store" or fl == "_DS_Store":
        print ("sorry")
        print(fl)
      else:
        images2 = os.path.join(file_name,fl)
        t = read_tensor_from_image_file(
            images2,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        # print(graph.get_operations())
        # print(input_name)
        input_operation = graph.get_operation_by_name(input_name)
        # print(input_operation)
        output_operation = graph.get_operation_by_name(output_name)
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
        results = np.squeeze(results)


        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)

        
        print(fl)
  

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  #writer.writeheader()
  #for i in t:
    writer.writerow({'imagename': fl,'Person': results[0],'Screen':results[1],'Print':results[2]})
end = time.time()
print(end - start)
    # except:
    #   print("failed")
    #   continue
    # for i in top_k:
    #   print(labels[i], results[i])