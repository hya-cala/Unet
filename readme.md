# ISBI2012 challenge with deeper Unet

This is the final project for CS420 SJTU 2020 Spring. The model is primarily inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Dataset

* Dataset was downloaded from [isbi challenge](http://brainiac2.mit.edu/isbi_challenge/), saved in "/dataset".


### Data Augmentation

* Only 25 images are available as training set, which is far from enough for deep fully convolutional network. Hence, data augmentation is a must for training.
* Simple augmentation includes flipping, rotation, shifting and shearing.
* [Elastic deformation](http://cognitivemedium.com/assets/rmnist/Simard.pdf) is an advanced method for data aumentation. However, this augmentation is quite slow and does not help the training. So I set it off as default.
* After augmentation, all images are padded to 704*704 by reflecting.


### Model

* Unlike some popular models on the Internent, we did not use any padding in convolution layers. becasuse we assumed that padding in hidden layers will only lead to noise. Instead, we use padded image of 714*714 as input data.
* Central cropping is applied in skipping layers.
* Dropout is applied before each downsamling to prevent from overfitting.

### Training

* Mini-batch gradient descent. Batch size = 2.
* Optimizer is Adam with a large learning_rate = 1e-4.
* accuracy ≈ 0.92, AUC ≈ 0.95, SSIM ≈ 0.76 on test set.


---

## Instruction

### Dependencies

* tensorflow == 2.2
* opencv-python
* numpy
* scikit-image

Your Python should be Python 3.

### pipeline.py

* elastic_transform can apply elastic deformation to a single image in OpenCV.
* TrainGenerator returns a generator as the pipeline to provide augmented data with certain batch size. It augmented the data and masks simultaneously. Data is augmented under 512*512, then padded to 706*706 by reflecting.
* ValidationDataPreparation returns the 25 unaugmented images as the validation set.
* TestDataPreparation returns 5 test images.

## model.py
* The deep convolutional network is implemented with Keras in Tensorflow 2.2.