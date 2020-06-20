# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:58:29 2020

@author: ZHANG
"""
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import time


def elastic_transform(image, alpha, sigma, alpha_affine):
    random_state = np.random.RandomState()
    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def TrainGenerator(batch_size,target_size = 704,deformation = False):
    SEED = 1 # Set the same seed for image_datagen and mask_datagen
    padding_pixels = (target_size-512)//2
    image_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=5,
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    shear_range=0.05,
                                    zoom_range=0.05,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode = 'reflect')
    mask_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=5,
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    shear_range=0.05,
                                    zoom_range=0.05,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode = 'reflect')
    image_generator = image_datagen.flow_from_directory('dataset/new train set', 
                                                        classes = ['train_img'],
                                                        class_mode = None, 
                                                        target_size=(512,512),
                                                        batch_size = batch_size,
                                                        color_mode='grayscale', 
                                                        seed = SEED,shuffle=False)
    mask_generator = mask_datagen.flow_from_directory('dataset/new train set', 
                                                        classes = ['train_label'],
                                                        class_mode = None, 
                                                        target_size=(512,512),
                                                        batch_size = batch_size,
                                                        color_mode='grayscale', 
                                                        seed = SEED,shuffle=False)
    generator = zip(image_generator, mask_generator)
    for (img,mask) in generator:
        if deformation == True:
            img_t = []
            mask_t = []
            for index in range(img.shape[0]):
                im_merge = np.concatenate((img[index], mask[index]), axis=2)
                im_merge_t = elastic_transform(im_merge, alpha = 3, sigma = 10, alpha_affine = 0)
                img_t.append(im_merge_t[...,0])
                temp = im_merge_t[...,1]
                mask_t.append(np.array(temp>0.5,dtype= np.float))
            img_t = np.pad(img_t, pad_width = ((0,0),(padding_pixels,padding_pixels),(padding_pixels,padding_pixels)), mode = 'reflect')
            yield (np.expand_dims(img_t,axis=3),np.expand_dims(mask_t,axis=3))
        else:
            img = np.pad(img, pad_width = ((0,0),(padding_pixels,padding_pixels),(padding_pixels,padding_pixels),(0,0)), mode = 'reflect')
            yield (img,mask)

def ValidationDataPreparation(target_size = 704,batch_size=1):
    padding_pixels = (target_size-512)//2
    image_datagen = ImageDataGenerator(rescale=1./255)
    mask_datagen = ImageDataGenerator(rescale=1./255)
    image_generator = image_datagen.flow_from_directory('dataset/new train set', 
                                                        classes = ['train_img'],
                                                        class_mode = None, 
                                                        target_size=(512,512),
                                                        batch_size = batch_size,
                                                        color_mode='grayscale',
                                                        shuffle=False)
    mask_generator = mask_datagen.flow_from_directory('dataset/new train set', 
                                                        classes = ['train_label'],
                                                        class_mode = None, 
                                                        target_size = (512,512),
                                                        batch_size = batch_size,
                                                        color_mode='grayscale',
                                                        shuffle=False)
    generator = zip(image_generator, mask_generator)
    for (img,mask) in generator:
        img = np.pad(img, pad_width = ((0,0),(padding_pixels,padding_pixels),(padding_pixels,padding_pixels),(0,0)), mode = 'reflect')
        yield(img,mask)
    
    
def TestDataPreparation(target_size = 704,batch_size = 1):
    padding_pixels = (target_size-512)//2
    image_datagen = ImageDataGenerator(rescale=1./255)
    mask_datagen = ImageDataGenerator(rescale=1./255)
    image_generator = image_datagen.flow_from_directory('dataset/new_test_set', 
                                                        classes = ['test_img'],
                                                        class_mode = None, 
                                                        target_size=(512,512),
                                                        color_mode='grayscale',
                                                        batch_size = batch_size,
                                                        shuffle=False)
    mask_generator = mask_datagen.flow_from_directory('dataset/new_test_set', 
                                                        classes = ['test_label'],
                                                        class_mode = None, 
                                                        target_size = (512,512),
                                                        color_mode='grayscale',
                                                        batch_size = batch_size,
                                                        shuffle=False)
    generator = zip(image_generator, mask_generator)
    for (img,mask) in generator:
        img = np.pad(img, pad_width = ((0,0),(padding_pixels,padding_pixels),(padding_pixels,padding_pixels),(0,0)), mode = 'reflect')
        yield(img,mask)