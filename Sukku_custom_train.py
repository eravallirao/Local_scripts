import random

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, LeakyReLU, Concatenate, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD
import cv2
import numpy as np
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.preprocessing.image import img_to_array
import os, sys
from keras.models import model_from_json
from keras.losses import cosine as cosine_loss
from utils import preprocess_input
from face_model import FaceModel
from keras.utils import plot_model
import keras
from keras_applications.imagenet_utils import _obtain_input_shape
import tensorflow as tf
import pickle
from alt_model_checkpoint.keras import AltModelCheckpoint
import argparse
from keras.utils import multi_gpu_model
import utils
from random import shuffle
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy as _accuracy
from binary_image_loader_softmax import PairImageLoader
from keras.regularizers import l2

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,3"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# def cosine_distance(vects):
#     y_true, y_pred = vects
#     def l2_normalize(x, axis):
#         norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
#         return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
#     y_true = l2_normalize(y_true, axis=-1)
#     y_pred = l2_normalize(y_pred, axis=-1)
#     return K.mean(1 - K.sum((y_true * y_pred), axis=-1, keepdims=True), keepdims=True)
def triplet_loss(y_true, y_pred, cosine = True, alpha = 0.2):
    embedding_size = K.int_shape(y_pred)[-1] // 3
    ind = int(embedding_size * 2)
    a_pred = y_pred[:, :embedding_size]
    p_pred = y_pred[:, embedding_size:ind]
    n_pred = y_pred[:, ind:]
    if cosine:
        positive_distance = 1 - K.sum((a_pred * p_pred), axis=-1)
        negative_distance = 1 - K.sum((a_pred * n_pred), axis=-1)
    else:
        positive_distance = K.sqrt(K.sum(K.square(a_pred - p_pred), axis=-1))
        negative_distance = K.sqrt(K.sum(K.square(a_pred - n_pred), axis=-1))
    loss = K.maximum(0.0, positive_distance - negative_distance + alpha)
    return loss


# def cosine_distance(vects):
#     y_true, y_pred = vects
#     def l2_normalize(x, axis):
#         norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
#         return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
#     y_true = l2_normalize(y_true, axis=-1)
#     y_pred = l2_normalize(y_pred, axis=-1)
#     return K.mean(1- (y_true * y_pred), axis=-1, keepdims=True)

def cosine_distance(vecs):
    x, y = vecs

    x_norm = K.sum(x * x, axis=1)
    y_norm = K.sum(y * y, axis=1)
    dot_prod = K.batch_dot(x,y,axes=1)

    scalar_prod_norm = K.sqrt(x_norm) * K.sqrt(y_norm)
    scalar_prod_norm = K.expand_dims(scalar_prod_norm,axis=1)
    out = dot_prod / (scalar_prod_norm + K.epsilon())
    return out


def cosine_distance_output_shape(shapes):
    return (None,1)

# def cosine_distance(vects):
#     x, y = vects
#     a = tf.matmul(tf.transpose(x), x)
#     x = K.l2_normalize(x, axis=-1)
#     y = K.l2_normalize(y, axis=-1)
#     c = tf.divide(a/)
#     return -K.mean(x * y, axis=-1, keepdims=True)

# def cos_dist_output_shape(shapes):
#     shape1, shape2 = shapes
#     return (shape1[0],1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):

    margin = 1
#    m2 = 0.75
#    m1 = 0.5
    m2 = 0.7
    m1 = 0.55

#    return K.mean((1 - y_true) * K.maximum(y_pred * y_pred - m2, 0) +
#                  y_true * K.maximum(m1 - y_pred * y_pred,0))
    return K.mean((1 - y_true) * K.maximum(m2 - (1 - y_pred * y_pred), 0) +
                  y_true * K.maximum(1 - y_pred * y_pred - m1, 0))

def initialize_bias(shape, dtype=None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def initialize_weights(shape, dtype=None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

class SiameseFaceModel(object):
    model_name = 'siamese-face-model'
    VERBOSE = 2

    def __init__(self):
        self.model = None

        self.labels = None
        self.config = None
        self.input_shape = None
        self.threshold = 0.1
        self.graph = None
        self.pretrained_model = self.load_pretrained_model()


    def img_to_encoding(self, image_path):
        # print('encoding: ', image_path)
        # if self.pretrained_model is None:
        #     self.pretrained_model = self.load_pretrained_model()

        image = cv2.imread(image_path, 1)
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        # input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        return input
        # return self.pretrained_model.predict(input)

    def load_model(self, model_dir_path):
        config_file_path = SiameseFaceModel.get_config_path(model_dir_path=model_dir_path)
        self.config = np.load(config_file_path,allow_pickle=True).item()
        self.labels = self.config['labels']
        self.input_shape = self.config['input_shape']
        self.threshold = self.config['threshold']

        if self.pretrained_model is None:
            self.pretrained_model = self.load_pretrained_model()
        self.model = self.create_network(input_shape=self.input_shape)
        weight_file_path = SiameseFaceModel.get_weight_path(model_dir_path)
        self.model.load_weights(weight_file_path)

    def load_model_pretrained(self, model_dir_path):
        if model_dir_path.endswith('/'):
            model_dir_path = model_dir_path[:-1]
        _config_file_path = model_dir_path + '/' + 'siamese-face-model-architecture.json'
        _weight_file_path = model_dir_path + '/' + 'siamese-face-model-weights.h5'
        with open(_config_file_path) as json_file:
            json_config = json_file.read()
        self.model = model_from_json(json_config)
        self.model.load_weights(_weight_file_path)
        print(self.model.summary())
        print('Model loaded from', _weight_file_path)

    def embedder(self,conv_feat_size):
        '''
        Takes the output of the conv feature extractor and yields the embeddings
        '''
        input = Input((conv_feat_size,), name = 'input')
        normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')
        x = Dense(1024)(input)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(512)(x)
        x = normalize(x)
        model = Model(input, x)
        return model

    def get_siamese_model(self,base_model):
        #this is actually version 2 of create_network
        inp_shape = K.int_shape(base_model.input)[1:]
        print('input shape',inp_shape)
        conv_feat_size = K.int_shape(base_model.output)[-1]
        print('conv feat shape', conv_feat_size)
        
        input_a = Input( inp_shape,  name='anchor')
        input_p = Input( inp_shape,  name='positive')
        input_n = Input( inp_shape,  name='negative')
        emb_model = self.embedder(conv_feat_size)
        output_a = emb_model(base_model(input_a))
        output_p = emb_model(base_model(input_p))
        output_n = emb_model(base_model(input_n))
        
        merged_vector = Concatenate(axis=-1)([output_a, output_p, output_n])
        model = Model(inputs=[input_a, input_p, input_n],
                    outputs=merged_vector)

        return model


    def fit_triplet(self, _dataset, model_dir_path, tensorboard_dir='./logs', epochs=20, batch_size=32, threshold=0.6):
        self.threshold = threshold
        # self.input_shape

        # self.pretrained_model = self.load_pretrained_model()
        self.model = self.get_siamese_model(self.pretrained_model)
        # self.model.load_weights('checkpoints/siamese-face-model-weights.h5')
        # if load_model is not None:
        #     self.model = load_model(load_model)
        # else:
        #     self.model = self.get_siamese_model(self.pretrained_model)


        names = []
        self.labels = dict()
        # for name in dataset.keys():
        #     names.append(name)
        #     self.labels[name] = len(self.labels)

        self.config = dict()
        self.config['input_shape'] = self.input_shape
        self.config['labels'] = self.labels
        self.config['threshold'] = self.threshold

        config_file_path = SiameseFaceModel.get_config_path(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

        weight_file_path = SiameseFaceModel.get_weight_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path, monitor='loss', verbose=SiameseFaceModel.VERBOSE, save_best_only=True, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1)
        tensorboard = TensorBoard(log_dir=tensorboard_dir)
        callbacks_list = [checkpoint,tensorboard, reduce_lr, early_stopping]

        # if _load_model is not None:
        #     self.model.load_weights(_load_model, by_name=True)

        adam = Adam(lr = 0.0003)
        # parallel_model = multi_gpu_model(self.model, 4)
        parallel_model = self.model

        parallel_model.compile(optimizer=adam, loss = triplet_loss)
        architecture_file_path = self.get_architecture_path(model_dir_path)
        with open(architecture_file_path, 'w+') as f:
            f.write(parallel_model.to_json())


        print(parallel_model.summary())


        print('Loading train and validation data')
        df_train, df_valid = utils.get_faces_df(_dataset)

        train_gen = PairImageLoader(df_train, preprocess_input, (224,224), batchSize = batch_size, flip=True, dataset_path=_dataset)
        valid_gen = PairImageLoader(df_valid, preprocess_input, (224,224), batchSize = batch_size, flip=True, dataset_path=_dataset)
        print('Data load done, starting train')

        parallel_model.fit_generator(train_gen, steps_per_epoch=len(train_gen),
                                 epochs=epochs, validation_data=valid_gen, validation_steps=len(valid_gen),
                                  callbacks=callbacks_list)

        parallel_model.save('checkpoints/triplet_final.h5')

    def create_siamese_network(self, input_shape):
        '''Base network to be shared (eq. to feature extraction).
        '''

        # return self.pretrained_model
        for layer in self.pretrained_model.layers[:15]:
            layer.trainable = False
        return self.pretrained_model
        # input = self.pretrained_model.output
        # normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')
        # x = Dense(4096, activation='sigmoid',
        #            kernel_regularizer=l2(1e-3),
        #            kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(input)


        # x = Dense(1024, activation='relu')(x)
        # x = Dropout(0.1)(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dropout(0.1)(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dropout(0.1)(x)
        # return Model(self.pretrained_model.input, input)

    def accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))

    def create_network(self, input_shape=None):
        # network definition
        input_shape = _obtain_input_shape((224, 224, 3),
                                    default_size=224,
                                    min_size=32,
                                    data_format=K.image_data_format(),
                                    require_flatten=True,
                                    weights='any')

        base_network = self.create_siamese_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        # rms = RMSprop(lr=0.0003, rho=0.8, epsilon=1e-6, decay=1e-7)
        # adam = Adam(lr=0.00001)

        '''Custom loss example'''
        # distance = Lambda(cosine_distance,
        #                   output_shape=cos_dist_output_shape)( [processed_a , processed_b] )
        # # output = Dense(1,activation='sigmoid')(distance)
        # # with tf.device('/cpu:0'):
        # model = Model([input_a, input_b], distance)
        # # model.compile(loss=contrastive_loss, optimizer=adam, metrics=[self.accuracy])
        distance = Lambda(cosine_distance, 
                  output_shape=cosine_distance_output_shape)([processed_a, processed_b])


        # fc1 = Dense(1024, kernel_initializer="glorot_uniform")(distance)
        # fc1 = Dropout(0.05)(fc1)
        # fc1 = Activation("relu")(fc1)
        # fc2 = Dense(1024, kernel_initializer="glorot_uniform")(fc1)
        # fc2 = Dropout(0.05)(fc2)
        # fc2 = LeakyReLU()(fc2)
        # pred = Dense(2, kernel_initializer="glorot_uniform")(fc2)
        # prediction = Activation("softmax")(pred)
        model = Model([input_a, input_b], distance)
        # model.compile(loss=binary_crossentropy, optimizer=adam, metrics=[self.accuracy])

        # print(model.summary())

        return model

    def create_pairs(self, dataset, names):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        num_classes = len(dataset)
        print('number of classes : ' + str(num_classes))
        pairs = []
        labels = []
        n = min([len(dataset[name]) for name in dataset.keys()]) -1
        for d in range(len(names)):
            name = names[d]
            x = dataset[name]
            for i in range(n):
                pairs += [[x[i], x[i + 1]]]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z1, z2 = x[i], dataset[names[dn]][i + 1]
                pairs += [[z1, z2]]
                labels += [0, 1]
        return np.array(pairs), np.array(labels)

    def create_pairs__(self, dataset, names):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        num_classes = len(dataset)
        print('number of classes : ' + str(num_classes))
        pairs = []
        labels = []
        n = min([len(dataset[name]) for name in dataset.keys()])
        n = 2
        for d in range(len(names)):
            name = names[d]
            x = dataset[name]
            if len(x) < 2:
                continue
				
            for i in range(len(x)):
                pairs += [[x[i], x[(i + 1) % n]]]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                neg_class_array = dataset[names[dn]]
                j = random.randrange(0,len(neg_class_array))
                z1, z2 = x[i], dataset[names[dn]][j]
                pairs += [[z1, z2]]
                labels += [1, 0]

        return np.array(pairs), np.array(labels)

    @staticmethod
    def get_config_path(model_dir_path):
        return model_dir_path + os.path.sep + SiameseFaceModel.model_name + '-config.npy'

    @staticmethod
    def get_weight_path(model_dir_path):
        return model_dir_path + os.path.sep + SiameseFaceModel.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_path(model_dir_path):
        return model_dir_path + os.path.sep + SiameseFaceModel.model_name + '-architecture.json'

    def load_pretrained_model(self):
        rest_model = FaceModel(input_shape=(224, 224, 3), model='resnet50', weights='./trained/facemodel/tf_resnet50.h5')
        face_descriptor_model = Model(input=rest_model.input, output=rest_model.layers[-2].output)
        self.graph = K.get_session().graph
        return face_descriptor_model

    def fit(self, _dataset, model_dir_path, tensorboard_dir='./logs', epochs=20, batch_size=32, threshold=0.6, save_to="."):
        self.threshold = threshold
        dataset = dict()
        # for _key in _dataset:
        #     if len(_dataset[_key]) > 1:
        #         dataset[_key] = _dataset[_key]
		
        # for name, feature in dataset.items():
        #     self.input_shape = feature[0].shape
        #     break

        # self.pretrained_model = self.load_pretrained_model()

        self.model = self.create_network(input_shape=self.input_shape)
        architecture_file_path = self.get_architecture_path(save_to)
        with open(architecture_file_path, 'w+') as f:
            f.write(self.model.to_json())

        names = []
        self.labels = dict()
        for name in dataset.keys():
            names.append(name)
            self.labels[name] = len(self.labels)

        self.config = dict()
        self.config['input_shape'] = self.input_shape
        self.config['labels'] = self.labels
        self.config['threshold'] = self.threshold

        config_file_path = SiameseFaceModel.get_config_path(model_dir_path=save_to)
        np.save(config_file_path, self.config)


        # parallel_model = multi_gpu_model(self.model,1)
        # parallel_model = self.model
        weight_file_path = SiameseFaceModel.get_weight_path(model_dir_path)
        # checkpoint = ModelCheckpoint(weight_file_path, monitor='loss', verbose=SiameseFaceModel.VERBOSE, save_best_only=True, mode='auto')
        alt_checkpoint = AltModelCheckpoint(save_to + '/' + 'model{epoch:02d}.h5', self.model)
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, verbose=2)
        early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=2)
        tensorboard = TensorBoard(log_dir=tensorboard_dir)
        callbacks_list = [alt_checkpoint,tensorboard, reduce_lr, early_stopping]

        # rms = RMSprop(lr=.00001)
        adam = Adam(lr=0.00001)
        # sgd = SGD(lr=0.0001)


        self.model.compile(loss=contrastive_loss, optimizer=adam, metrics=[self.accuracy])
        print(self.model.summary())

        df_train, df_valid = utils.get_faces_df(_dataset)

        train_gen = PairImageLoader(df_train, preprocess_input, (224,224), batchSize = batch_size, flip=True, dataset_path=_dataset)
        valid_gen = PairImageLoader(df_valid, preprocess_input, (224,224), batchSize = batch_size, flip=False, dataset_path=_dataset)
        # valid_gen = [valid_gen[:0], valid_gen[:,1]]
        print('Data load done, starting train')

        self.model.fit_generator(train_gen, steps_per_epoch=len(train_gen),
                                 epochs=epochs, validation_data=valid_gen, validation_steps=len(valid_gen),
                                  callbacks=callbacks_list, workers=12, use_multiprocessing=True)

        self.model.save(save_to + "/" + "final.h5")
    def fit_multi_gpu(self, _dataset, model_dir_path, tensorboard_dir='./logs', epochs=5, batch_size=32, threshold=0.6):
        self.threshold = threshold
        dataset = dict()
        for _key in _dataset:
            if len(_dataset[_key]) > 1:
                dataset[_key] = _dataset[_key]
		
        for name, feature in dataset.items():
            self.input_shape = feature[0].shape
            break

        self.pretrained_model = self.load_pretrained_model()

        self.model = self.create_network(input_shape=self.input_shape)
        architecture_file_path = self.get_architecture_path(model_dir_path)
        with open(architecture_file_path, 'w+') as f:
            f.write(self.model.to_json())

        names = []
        self.labels = dict()
        for name in dataset.keys():
            names.append(name)
            self.labels[name] = len(self.labels)

        self.config = dict()
        self.config['input_shape'] = self.input_shape
        self.config['labels'] = self.labels
        self.config['threshold'] = self.threshold

        config_file_path = SiameseFaceModel.get_config_path(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

        weight_file_path = SiameseFaceModel.get_weight_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path, verbose=SiameseFaceModel.VERBOSE)
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1)
        tensorboard = TensorBoard(log_dir=tensorboard_dir)
        callbacks_list = [checkpoint,tensorboard, early_stopping, reduce_lr]
        t_x, t_y = self.create_pairs(dataset, names)

        print('data set pairs: ', t_x.shape)
        _multi_gpu_model = multi_gpu_model(self.model,4)
        # rms = RMSprop(lr=5e-3, rho=0.8, epsilon=1e-6, decay=1e-5)
        adam = Adam(lr=0.001)


        _multi_gpu_model.compile(loss=contrastive_loss, optimizer=adam, metrics=[self.accuracy])
        print(_multi_gpu_model.summary())

        _multi_gpu_model.fit([t_x[:, 0], t_x[:, 1]], t_y,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.1,
                       callbacks=callbacks_list)
        _multi_gpu_model.save_weights(weight_file_path)

        # _multi_gpu_model.save('exported/siamese_trained.h5')

    def merge_and_export(self,output_dir):
        # plot_model(self.pretrained_model,'model1.png')
        # plot_model(self.model, 'model2.png')
        # self.model.save('merged/siamese_face.h5')
        self.pretrained_model = self.load_pretrained_model()
        self.pretrained_model.save('exported/exported_pretrained.h5')

    def load_dataset(self, file_name):
        with open(file_name,'rb') as f:
            return pickle.load(f)


    def findCosineDistance(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def compare_images(self,image1, image2):
        _im1 = self.img_to_encoding(image1)
        _im2 = self.img_to_encoding(image2)

        _dist = self.compare(_im1,_im2)
        print(_dist)

    def compare(self, image1_enconding, image2_encoding):

        # Step 2: Compute distance with identity's image (â‰ˆ 1 line)
        input_pairs = [[image1_enconding, image2_encoding]]
        input_pairs = np.array(input_pairs)
        dist = np.average(self.model.predict([input_pairs[:, 0], input_pairs[:, 1]]), axis=-1)[0]

        return dist

def preprocess_image(image):
    resized_image = cv2.resize(image,(168,224))
    image_padded = cv2.copyMakeBorder( resized_image, 0, 0, 28, 28, cv2.BORDER_CONSTANT)
    _input = img_to_array(image_padded)
    _input = np.expand_dims(_input, axis=0)
    _input = utils.preprocess_input(_input, version=2)
    _input = np.squeeze(_input)

    return _input


def pre_process_dataset(dataset):
    counter = 0
    for key in dataset:
        images = dataset[key]
        for i, _image in enumerate(images):
            out_image = preprocess_image(_image)
            images[i] = out_image
            counter += 1
            if counter % 5000 == 0:
                print('preocessed', counter)
        dataset[key] = images
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Face Comparison')

    parser.add_argument('--phase', type=str, required=True,
                        help='Train or test phase')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset file')
    parser.add_argument('--epoch', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--thresh', type=float, required=False, default=.6,
                        help='Threshold for face comparison, lower means better match, default value is 0.6')
    parser.add_argument('--batch', type=int, required=False, default=32,
                        help='Batch size for training')
    parser.add_argument('--load_model_from', type=str, required=False, default=None,
                        help='Load model from file for futher fine tunning')
    parser.add_argument('--save_model', type=str, required=False, default=None,
                        help='Load model from file for futher fine tunning')
    parser.add_argument('--multi_gpu', type=bool, required=False, default=False,
                        help='multi or single gpu')
    # parser.add_argument('--image1', type=str, required=True,
    #                     help='Train or test phase')
    # parser.add_argument('--image2', type=str, required=True,
    #                     help='Train or test phase')

    args = parser.parse_args()

    if args.phase == 'train':
        fnet = SiameseFaceModel()

        model_dir_path = args.load_model_from
        if args.load_model_from is not None:
            fnet.load_model_pretrained(args.load_model_from)
        # dataset_file = args.dataset

        # dataset = fnet.load_dataset(dataset_file)
        # dataset = dict(list(dataset.items())[len(dataset)//3:])
        # dataset = pre_process_dataset(dataset)

        # _list = list(dataset.items())
        # shuffle(_list)
        # dataset = dict(_list)

        # if args.multi_gpu == True:
        #     fnet.fit_multi_gpu(_dataset=dataset, model_dir_path=model_dir_path,epochs=args.epoch,batch_size=args.batch, threshold=args.thresh)
        # else:
        fnet.fit(_dataset=args.dataset, model_dir_path=model_dir_path,epochs=args.epoch,batch_size=args.batch, threshold=args.thresh, save_to=args.save_model)

        # keras.backend.set_learning_phase(0)
        # fnet.load_model(model_dir_path)
        # fnet.merge_and_export('out')
    if args.phase == 'test':
        fnet = SiameseFaceModel()

        model_dir_path = './test'
        fnet.load_model(model_dir_path)
        fnet.compare_images(args.image1,args.image2)

    if args.phase == 'export':
        fnet = SiameseFaceModel()
        fnet.merge_and_export('out')    



if __name__ == '__main__':
    main()



# gs://fr-arcface