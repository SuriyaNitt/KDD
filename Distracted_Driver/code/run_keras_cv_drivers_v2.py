# -*- coding: utf-8 -*-

import numpy as np
import shutil

np.random.seed(2016)

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import statistics
import time
from shutil import copy2
import warnings
import random

warnings.filterwarnings("ignore")

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from scipy.misc import imread, imresize, imshow
from keras.regularizers import l1, l2, l1l2, activity_l1, activity_l2, activity_l1l2
from keras.utils.visualize_util import plot

from keras import backend as K
from subprocess import call

use_cache = 1


def show_image(im, name='image'):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# color_type = 1 - gray
# color_type = 3 - RGB
def get_im_cv2(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
    return resized


def get_im_cv2_mod(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)
    # Reduce size
    rotate = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rotate, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
    return resized


def get_driver_data():
    dr = dict()
    clss = dict()
    path = os.path.join('..', 'input', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
        if arr[0] not in clss.keys():
            clss[arr[0]] = [(arr[1], arr[2])]
        else:
            clss[arr[0]].append((arr[1], arr[2]))
    f.close()
    return dr, clss


def load_train(img_rows, img_cols, color_type=1):
    X_train = []
    X_train_id = []
    y_train = []
    driver_id = []
    start_time = time.time()
    driver_data, dr_class = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('..', 'input', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            # img = get_im_cv2(fl, img_rows, img_cols, color_type)
            img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, X_train_id, driver_id, unique_drivers


def load_test(img_rows, img_cols, color_type=1):
    print('Read test images')
    path = os.path.join('..', 'input', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    start_time = time.time()
    thr = math.floor(len(files) / 10)
    for fl in files:
        flbase = os.path.basename(fl)
        # img = get_im_cv2(fl, img_rows, img_cols, color_type)
        img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def save_model(model, arch_path, weights_path):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(arch_path, 'w').write(json_string)
    model.save_weights(weights_path, overwrite=True)


def read_model(arch_path, weights_path):
    model = model_from_json(open(arch_path).read())
    model.load_weights(weights_path)
    return model


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


def save_useful_data(predictions_valid, valid_ids, model, info):
    result1 = pd.DataFrame(predictions_valid, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(valid_ids, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir(os.path.join('subm', 'data')):
        os.mkdir(os.path.join('subm', 'data'))
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    # Save predictions
    pred_file = os.path.join('subm', 'data', 's_' + suffix + '_train_predictions.csv')
    result1.to_csv(pred_file, index=False)
    # Save model
    json_string = model.to_json()
    model_file = os.path.join('subm', 'data', 's_' + suffix + '_model.json')
    open(model_file, 'w').write(json_string)
    # Save code
    cur_code = os.path.realpath(__file__)
    code_file = os.path.join('subm', 'data', 's_' + suffix + '_code.py')
    copy2(cur_code, code_file)


def read_and_normalize_train_data(img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(
        color_type) + '_rotated.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, train_id, driver_id, unique_drivers = load_train(img_rows, img_cols, color_type)
        cache_data((train_data, train_target, train_id, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, train_id, driver_id, unique_drivers) = restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))

    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id, driver_id, unique_drivers


def read_and_normalize_test_data(img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(
        color_type) + '_rotated.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))
    # test_data = test_data.swapaxes(3, 1)
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1 / nfolds)
    return a.tolist()


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data)
    target = np.array(target)
    index = np.array(index)
    return data, target, index


def load_partial_weights(model, file_path):
    """Load partial layer weights from a HDF5 save file.
        """
    import h5py
    f = h5py.File(file_path, mode='r')

    if hasattr(model, 'flattened_layers'):
        # support for legacy Sequential/Merge behavior
        flattened_layers = model.flattened_layers
    else:
        flattened_layers = model.layers

    if 'nb_layers' in f.attrs:

        for k in range(len(flattened_layers)):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            flattened_layers[k].set_weights(weights)

    else:
        print('nb_layers attribute missing in given file')
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        if len(layer_names) != len(flattened_layers):
            print('You are trying to load a weight file '
                  'containing ' + str(len(layer_names)) +
                  ' layers into a model with ' +
                  str(len(flattened_layers)) + ' layers.')

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        layer_count = 0
        print layer_names
        print len(flattened_layers)
        model_k = 0
        for k, name in enumerate(layer_names):
            # Suriya debug
            print k
            layer_count += 1

            if layer_count >= len(flattened_layers):
                continue
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]
                if model_k >= len(flattened_layers):
                    continue
                layer = flattened_layers[model_k]
                print('model_layer: {}, saved_layer: {}'.format(layer.name, name))
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                if (len(weight_values) != len(symbolic_weights)) and (layer.name != name):
                    print('Layer #' + str(k) +
                                    ' (named "' + layer.name +
                                    '" in the current model) was found to '
                                    'correspond to layer ' + name +
                                    ' in the save file. '
                                    'However the new layer ' + layer.name +
                                    ' expects ' + str(len(symbolic_weights)) +
                                    ' weights, but the saved weights have ' +
                                    str(len(weight_values)) +
                                    ' elements.')
                    model_k += 1
                    layer = flattened_layers[model_k]
                    print('Layers after forwarding through the model')
                    print('model_layer: {}, saved_layer: {}'.format(layer.name, name))
                    symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                    if (len(weight_values) != len(symbolic_weights)) and (layer.name != name):
                        print('Layer #' + str(k) +
                              ' (named "' + layer.name +
                              '" in the current model) was found to '
                              'correspond to layer ' + name +
                              ' in the save file. '
                              'However the new layer ' + layer.name +
                              ' expects ' + str(len(symbolic_weights)) +
                              ' weights, but the saved weights have ' +
                              str(len(weight_values)) +
                              ' elements.')

                weight_value_tuples += zip(symbolic_weights, weight_values)

            model_k += 1

        K.batch_set_value(weight_value_tuples)

    f.close()


def create_model_v1(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')
    return model


def create_model_v2(img_rows, img_cols, color_type=1):
    model1 = Sequential()
    model1.add(Convolution2D(32, 3, 3, border_mode='same', trainable=True, init='he_normal',
                             input_shape=(color_type, img_rows, img_cols)))
    model1.add(BatchNormalization(epsilon=1e-06, mode=2, axis=1, momentum=0.9, weights=None, beta_init='zero',
                                 gamma_init='one'))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))

    model2 = Sequential()
    model2.add(Convolution2D(32, 5, 5, border_mode='same', trainable=True, init='he_normal',
                             input_shape=(color_type, img_rows, img_cols)))
    model2.add(BatchNormalization(epsilon=1e-06, mode=2, axis=1, momentum=0.9, weights=None, beta_init='zero',
                                 gamma_init='one'))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    model = Sequential()
    model.add(Merge([model1, model2], mode='concat', concat_axis=1))

    model.add(Convolution2D(64, 3, 3, border_mode='same', trainable=True, init='he_normal'))
    model.add(BatchNormalization(epsilon=1e-06, mode=2, axis=1, momentum=0.9, weights=None, beta_init='zero',
                                 gamma_init='one'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3, border_mode='same', trainable=True, init='he_normal'))
    model.add(BatchNormalization(epsilon=1e-06, mode=2, axis=1, momentum=0.9, weights=None, beta_init='zero',
                                 gamma_init='one'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    '''
    model.add(Convolution2D(256, 3, 3, border_mode='same', trainable=True, init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Convolution2D(512, 3, 3, border_mode='same', trainable=True, init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    '''

    #load_partial_weights(model, './cache/stack_color/conv_5.h5')
    #model.layers.pop()

    model.add(Flatten())

    model.add(Dense(1024, init='he_normal', bias=True, trainable=True))
    model.add(BatchNormalization(epsilon=1e-06, mode=2, axis=1, momentum=0.9, weights=None, beta_init='zero',
                                 gamma_init='one'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    #model.add(Dense(4096, init='he_normal', bias=True, trainable=True))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.4))

    #model.add(Dense(1024, init='he_normal', bias=True, trainable=True))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')

    plot(model, to_file='model.png')

    return model


def vgg_std16_model(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('/home/suriya/Documents/Machine_Learning/vgg16_weights.h5')

    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def create_model_v3(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(Convolution2D(16, 5, 5, border_mode='valid', init='he_normal', subsample=(4,4),
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 5, 5, border_mode='valid', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 5, 5, border_mode='valid', init='he_normal'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')
    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation(nfolds=10):
    # input image dimensions
    img_rows, img_cols = 224, 224 #64, 64
    # color type: 1 - grey, 3 - rgb
    color_type_global = 3
    batch_size = 16
    nb_epoch = 50
    random_state = 51
    restore_from_last_checkpoint = 1
    model_version = 1

    train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols,
                                                                                                  color_type_global)
    test_data, test_id = train_data[0], train_id[0]  # read_and_normalize_test_data(img_rows, img_cols, color_type_global)
    #test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)
    model = Sequential()
    if model_version == 1:
        model = create_model_v3(img_rows, img_cols, color_type_global)
        #model = vgg_std16_model(img_rows, img_cols, color_type_global)
    elif model_version == 2:
        model = create_model_v2(img_rows, img_cols, color_type_global)

    yfull_train = dict()
    yfull_test = []
    kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    for train_drivers, test_drivers in kf:
        unique_list_train = [unique_drivers[i] for i in train_drivers]
        X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
        unique_list_valid = [unique_drivers[i] for i in test_drivers]
        X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

        num_fold += 1
        print('\nStart KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        print('Train drivers: ', unique_list_train)
        print('Test drivers: ', unique_list_valid)

        x_train = []
        x_valid = []

        if model_version ==2:
            x_train = [X_train, X_train]
            x_valid = [X_valid, X_valid]
        elif model_version == 1:
            x_train = X_train
            x_valid = X_valid



        kfold_weights_path = os.path.join('cache', 'weights_kfold_' + str(num_fold) + '.h5')
        if not os.path.isfile(kfold_weights_path) or restore_from_last_checkpoint == 0:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=1, verbose=0),
                ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
            ]
            model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                      shuffle=True, verbose=1, validation_data=(x_valid, Y_valid),
                      callbacks=callbacks)
        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)

        # score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
        # print('Score log_loss: ', score[0])

        predictions_valid = model.predict(x_valid, batch_size=batch_size, verbose=1)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score * len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        test_data_model2 = []

        if model_version == 2:
            test_data_model2 = [test_data, test_data]
        elif model_version == 1:
            test_data_model2 = test_data

        # Store test predictions
        test_prediction = model.predict(test_data_model2, batch_size=batch_size, verbose=1)
        yfull_test.append(test_prediction)

    score = sum_score / len(train_data)
    print("Log_loss train independent avg: ", score)

    predictions_valid = get_validation_predictions(train_data, yfull_train)
    score1 = log_loss(train_target, predictions_valid)
    if abs(score1 - score) > 0.0001:
        print('Check error: {} != {}'.format(score, score1))

    print(
    'Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(score, img_rows, img_cols, nfolds, nb_epoch))
    info_string = 'loss_' + str(score) \
                  + '_clr_' + str(color_type_global) \
                  + '_bat_sz' + str(batch_size) \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(nfolds) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    # test_res = merge_several_folds_geom(yfull_test, nfolds)
    create_submission(test_res, test_id, info_string)
    save_useful_data(predictions_valid, train_id, model, info_string)


def run_single():
    # input image dimensions
    img_rows, img_cols = 64, 64
    color_type_global = 1
    batch_size = 32
    nb_epoch = 50
    random_state = 51

    train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols,
                                                                                                  color_type_global)
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)

    yfull_test = []
    unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p035', 'p041', 'p042', 'p045', 'p047',
                         'p049', 'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p075', 'p081']
    X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
    unique_list_valid = ['p024', 'p026', 'p039', 'p072']
    X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

    print('Start Single Run')
    print('Split train: ', len(X_train))
    print('Split valid: ', len(X_valid))
    print('Train drivers: ', unique_list_train)
    print('Valid drivers: ', unique_list_valid)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ]
    model = create_model_v1(img_rows, img_cols, color_type_global)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)

    # score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
    # print('Score log_loss: ', score[0])

    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    score = log_loss(Y_valid, predictions_valid)
    print('Score log_loss: ', score)

    # Store test predictions
    test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
    yfull_test.append(test_prediction)

    print('Final log_loss: {}, rows: {} cols: {} epoch: {}'.format(score, img_rows, img_cols, nb_epoch))
    info_string = 'loss_' + str(score) \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_ep_' + str(nb_epoch)

    full_pred = model.predict(train_data, batch_size=batch_size, verbose=1)
    score = log_loss(train_target, full_pred)
    print('Full score log_loss: ', score)

    test_res = merge_several_folds_mean(yfull_test, 1)
    create_submission(test_res, test_id, info_string)
    save_useful_data(full_pred, train_id, model, info_string)

def dump_classified_images():
    if not os.path.isdir(os.path.join('..', 'output')):
        os.mkdir(os.path.join('..', 'output'))
    for i in range(10):
        dir_name = 'c' + str(i)
        if not os.path.isdir(os.path.join('..', 'output', dir_name)):
            os.mkdir(os.path.join('..', 'output', dir_name))

    best_subm = \
        os.path.join('subm',
                     'submission_loss_0.44449654537_clr_3_bat_sz16_r_64_c_64_folds_13_ep_50_2016-06-21-14-10.csv')
    print('Dumping the classified images')
    f = open(best_subm, 'r')
    line = f.readline()
    count = 0
    while(1):
        count += 1
        line = f.readline()
        if line == '':
            break
        arr = line.split(',')
        img = arr[-1]
        arr = arr[:-1]
        arr = np.array(arr)
        folder = 'c' + str(arr.argmax())
        parent_dir = '/home/suriya/Documents/Kaggle/Distracted_Driver'
        copy_cmd = 'cp ' + parent_dir + '/input/test/' + img.strip() \
                   + ' ' + parent_dir + '/output/' + folder + '/' + img.strip()
        print str(count / 79726.0 * 100.0) + '%'
        #call(copy_cmd, cwd=parent_dir)
        shutil.copyfile(os.path.join('..', 'input', 'test', img.strip()),
                        os.path.join('..', 'output', folder, img.strip()))



if __name__ == '__main__':
    #dump_classified_images()
    run_cross_validation(13)
    # run_single()
