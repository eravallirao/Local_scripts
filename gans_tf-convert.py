import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import os
import argparse

# ap=argparse.ArgumentParser()
# ap.add_argument('-m','--model_name',type=str,default='',help='path')
# args=vars(ap.parse_args())
basedir = os.path.dirname(__file__)
model_name = "gen_2"
# model_path =args['model_name']

model = tf.keras.models.load_model("/Users/sukshi/Downloads/GANs_for_QR/GANs_for_production/with_weig_gen_2.h5")

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# Change your model name here
version = "1/"
export_path = "../models/" + model_name + "/serving/" + version
path = os.path.join(basedir, export_path)
# path = 'exported/aadhar_back_digi_spoof/serving/1/'

#  Make sure you give the correct input name
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
            sess,
            path,
            inputs={'input_1': model.input},
            outputs={t.name:t for t in model.outputs})
print("MODEL : ",model_name)
print(model.outputs)