import json

import numpy as np

import keras
print(keras.__version__)
from keras.models import Model
import keras.layers as layers
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from modelanddata import *

params = "./params.json"
#Load the hyperparameter file. This is a json file.
with open(params) as json_file:  
    params = json.load(json_file)

#Set the path of train and test. Also set the path to save the model 

outing = "./"
weightfile = "./NIMA.hdf5"


#Define the base model for loading the structure of the mobilenet architecture.
base_model = MobileNet((224, 224, 3), alpha=1, include_top=False, pooling='avg', weights=None)
#Load the weights in the model from the NIMA architecture. We only require the convolution layers
#of the NIMA architecture
base_model.load_weights(weightfile, by_name=True)

#Define a model cut till the conv_pw_12_relu layer.
modelcut = build_bottleneck_model(base_model, 'conv_pw_13_relu')

#Add extra Depthwise conv BLocks
interimoutput =depthwise_conv_block(modelcut.output, 2048, 1, 1, strides=(2, 2), block_id=14)

#Do Global Average pooling as intended in the NIMA paper.
interimoutput = GlobalAveragePooling2D()(interimoutput)

#The Dropout as in the NIMA paper.
interimoutput = Dropout(.35)(interimoutput)

#Dense layers for the final output.
interimoutput = Dense(32, activation='relu', name = "Dense_1")(interimoutput)
interimoutput = Dense(16, activation='relu', name = "Dense_2")(interimoutput)
interimoutput = Dense(8, activation='relu', name = "Dense_3")(interimoutput)

#Final output
finaloutput = Dense(2, activation='softmax', name="Logits")(interimoutput)

#Define the model
model = Model(base_model.input, finaloutput, name='Modified-NIMA-Architecture')
print("The Model Summary is:")
print(model.summary())

# Freeze the layers except the last 4 layers
for layer in model.layers[:-9]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in model.layers:
    print(layer, layer.trainable)

train_generator, validation_generator = makedatagenerators(params)

# Compile the model
model.compile(loss=params['loss'],
              optimizer=params['optimizer'],
              metrics=['accuracy'])

# Train the model
model.fit_generator(
      train_generator,
      steps_per_epoch=params['steps_per_epoch'],
      epochs=params['epochs'],
      validation_data=validation_generator,
      validation_steps=params['validation_steps'])

model.save("modified-NIMA")
reloadmodel = keras.models.load_model('modified-NIMA')
print("Summar of the saved model for verification :")
reloadmodel.summary()
