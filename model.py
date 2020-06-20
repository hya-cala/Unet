# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:58:29 2020

@author: ZHANG
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Cropping2D, MaxPooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, AUC

def unet(padding = 'valid'):#The final model
    if padding == 'valid':
        inputs = Input((704,704,1))
    else:
        inputs = Input((512,512,1))
    conv1 = Conv2D(filters = 64, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(inputs)
    conv1 = Conv2D(filters = 64, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(conv1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = Conv2D(filters = 64, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(pool1)
    conv2 = Conv2D(filters = 128, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(conv2)
    conv2 = Conv2D(filters = 128, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(conv2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(filters = 256, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(pool2)
    conv3 = Conv2D(filters = 256, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(filters = 512, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(pool3)
    conv4 = Conv2D(filters = 512, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(filters = 1024, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(pool4)
    conv5 = Conv2D(filters = 1024, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(conv5)

    up6 = UpSampling2D(size = (2,2))(conv5)
    merge6 = Concatenate(axis = 3)([Cropping2D(cropping=(4,4))(drop4), up6])
    conv6 = Conv2D(filters = 512, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(merge6)
    conv6 = Conv2D(filters = 512, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(conv6)

    up7 = UpSampling2D(size = (2,2))(conv6)
    merge7 =Concatenate(axis = 3)([Cropping2D(cropping=(16,16))(drop3), up7])
    conv7 = Conv2D(filters = 256, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(merge7)
    conv7 = Conv2D(filters = 256, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(conv7)

    up8 = UpSampling2D(size = (2,2))(conv7)
    merge8 =Concatenate(axis = 3)([Cropping2D(cropping=(40,40))(drop2), up8])
    conv8 = Conv2D(filters = 128, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(merge8)
    conv8 = Conv2D(filters = 128, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(conv8)
    conv8 = Conv2D(filters = 128, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(conv8)

    up9 = UpSampling2D(size = (2,2))(conv8)
    merge9 = Concatenate(axis = 3)([Cropping2D(cropping=(92,92))(drop1), up9])
    conv9 = Conv2D(filters = 64, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(merge9)
    conv9 = Conv2D(filters = 64, kernel_size = (3,3), padding = padding, activation = 'relu', kernel_initializer = 'VarianceScaling')(conv9)
    conv10 = Conv2D(filters = 1, kernel_size = (1,1), padding = padding, activation = 'sigmoid', kernel_initializer = 'VarianceScaling')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [BinaryAccuracy(),AUC()])

    return model

def unet_608():# The swallow model
    inputs = Input((608,608,1))

    conv1 = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', kernel_initializer = 'VarianceScaling')(inputs)
    conv1 = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', kernel_initializer = 'VarianceScaling')(conv1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', kernel_initializer = 'VarianceScaling')(pool1)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(filters = 256, kernel_size = (3,3), activation = 'relu', kernel_initializer = 'VarianceScaling')(pool2)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(filters = 512, kernel_size = (3,3), activation = 'relu', kernel_initializer = 'VarianceScaling')(pool3)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(filters = 1024, kernel_size = (3,3), activation = 'relu', kernel_initializer = 'VarianceScaling')(pool4)

    up6 = UpSampling2D(size = (2,2))(conv5)
    merge6 = Concatenate(axis = 3)([Cropping2D(cropping=(2,2))(drop4), up6])
    conv6 = Conv2D(filters = 512, kernel_size = (3,3), activation = 'relu', kernel_initializer = 'VarianceScaling')(merge6)

    up7 = UpSampling2D(size = (2,2))(conv6)
    merge7 =Concatenate(axis = 3)([Cropping2D(cropping=(8,8))(drop3), up7])
    conv7 = Conv2D(filters = 256, kernel_size = (3,3), activation = 'relu', kernel_initializer = 'VarianceScaling')(merge7)

    up8 = UpSampling2D(size = (2,2))(conv7)
    merge8 =Concatenate(axis = 3)([Cropping2D(cropping=(20,20))(drop2), up8])
    conv8 = Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', kernel_initializer = 'VarianceScaling')(merge8)

    up9 = UpSampling2D(size = (2,2))(conv8)
    merge9 = Concatenate(axis = 3)([Cropping2D(cropping=(44,44))(drop1), up9])
    conv9 = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', kernel_initializer = 'VarianceScaling')(merge9)
    conv9 = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', kernel_initializer = 'VarianceScaling')(conv9)
    conv10 = Conv2D(filters = 1, kernel_size = (1,1), activation = 'sigmoid', kernel_initializer = 'VarianceScaling')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [BinaryAccuracy(),AUC()])

    return model

def unet_padding(input_size = (512,512,1)):# Model downloaded from the Internent.
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = Concatenate(axis = 3)([drop4, up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 =Concatenate(axis = 3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 =Concatenate(axis = 3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = Concatenate(axis = 3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [BinaryAccuracy(),AUC()])

    return model