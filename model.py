import numpy as np

import pandas as pd   #not apparently needed, but kept around for future possibilities

import matplotlib    #this is really for debugging only, not actually used if not debugging image preprocessing
matplotlib.use('agg')   #because Tkinter is apparently broken
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.models import load_model

from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

from keras.optimizers import Adam

from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras import backend as K

import tensorflow as tf

from scipy.misc import imresize
from tqdm import tqdm

#others imported sklearn, but this is unneeded with the implementation of keras' image.flow_from_directory

import os

import itertools

from unet_build import *


##VERSIONS: Tensorflow 1.2.0; Keras 2.0.6
#Should work with later versions just fine, though some minor changes to imports may be useful.
#The architecture implemented here is a 256x256 U-Net, though I may add support for deeper U-Nets if this works


CNN_name = 'test'  #name the CNN for saving weights (do if training multiple or different ones)


#SUPER IMPORTANT - DEFINE WHICH GPU TO USE, IF RELEVANT
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # or '1' or whichever GPU is available on your machine
#use nvidia-smi to see which devices are currently in use by one person or another, and use the one not in use.



#Define seed to be used (for reproducibility and to make sure the masks have the same transforms
seed = 999  #can make this a randomly generated number if don't care about reproducibility, but the variable still needs to be assigned for consistent transformations



#Define training directories (raw training images and their corresponding masks)
train_image_dir = "input/train/"
train_mask_dir = "input/train_masks/"

#Define testing directories (raw testing/validation images and their corresponding masks)
test_image_dir = "input/test/"
test_mask_dir = "input/test_masks/"


#Images should be PNG or JPG, may handle TIF if in a later version of Keras

#NOTE: IMAGES MUST BE IN A SUBDIRECTORY OF THESE DIRECTORIES
#Example file structure (the subdirectory '0' can be anything, and the directories can be in different places according to how you've defined them, but the directories you define must have that subdirectory)

'''
        input/
            train/
                0/
                    0.png
                    1.png
                    2.png
                    3.png
                    ...
            train_mask/
                0/
                    0.png
                    1.png
                    2.png
                    3.png
                    ...
            test/
                0/
                    0.png
                    1.png
                    2.png
                    3.png
                    ...
            test_mask/
                0/
                    0.png
                    1.png
                    2.png
                    3.png
                    ...
'''


#Define values for the CNN implementation
resizeHeight = 512  #height of image to be entered into CNN
resizeWidth = 512  #width of image to be entered into CNN
numChannels = 1   #number of channels of image to be entered into CNN; 1 is greyscale, 3 is rgb  --should always be set to 1, because the CNN can't handle 3 as it is currently.

batchSize = 4   #Define batch sizes (how many images will be loaded into memory at a time) - same for both training and validation in this implementation

trainEpochSteps = 32  #number of training steps per epoch (how many batches to take out per epoch), typically the number of training images you have divided by the number of images in each batch to cover all the images
valEpochSteps = 8  #number of validation steps per epoch (how many batches to take out per epoch), typically the number of validation images you have divided by the number of images in each batch to cover all the images

numEpochs = 4000  #one of the most important things: how many epochs should the network go through before ending?

learningRate = 1e-4  #this is almost 1e-4 for the Adam optimizer, though other optimizers may not even use a learning rate value or may use different ones.

lossFunc = 'binary_crossentropy'  #can be dice_coef_loss, mean_iou, or binary_crossentropy (AS STRINGS)


#Define values for the run
usePrebuiltModel = False   #Don't build the U-Net model from scratch (do this if you only have the weights, otherwise set this to false to build the model from scratch

doTraining = True   #Train the model. If you don't do this, it will cause errors or strange behavior if you don't preload weights from somewhere.

usePremadeWeights = False   #If using premade weights, then load them instead of the model's weights or instead of training

loadModelFile = 'modelFiles/newFullModel.h5'   #Define where the model should be loaded from if using a prebuilt one

saveModelFile = 'modelFiles/fullModel.h5'   #Define where the model should be saved if doing training

loadWeightFile = 'modelFiles/weights/fullWeights.h5'    #Define where to load the weight file from if using premade weights

saveWeightDir = 'modelFiles/weights/'    #Define where to save the progressive weights generated if training




#Define data augmentation parameters for training; not using shear because it's better for image classification rather than for actual segmentation, but it may be useful for future kinds of images, so it's just commented out
data_aug_param = dict(rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     rescale=1./255,
                     #shear_range = 0.2,
                     horizontal_flip = True,
                     vertical_flip = True)

##version with fewer parameters for debugging/faster running
#data_aug_param = dict(rotation_range=90.,
                     #zoom_range=0.2,
                     #rescale=1./255,
                     #horizontal_flip = True) 



#Definitions of various things for image generator
inputTargetSize = (resizeHeight, resizeWidth)

if numChannels==1:
    colorMode = 'grayscale'
elif numChannels==3:
    colorMode = 'rgb'
else:
    print 'Not sure how to handle this... defaulting to RGB for now.'
    colorMode = 'rgb'



  

#Define generators for the training images and mask images, making sure to apply the same augmentations to each and to confirm their final outputs coincide (augmented mask matches augmented image output)
train_image_datagen = ImageDataGenerator(**data_aug_param)
train_image_generator = train_image_datagen.flow_from_directory(train_image_dir, class_mode=None, seed=seed, color_mode=colorMode, target_size=inputTargetSize, batch_size=batchSize)

train_mask_datagen = ImageDataGenerator(**data_aug_param)
train_mask_generator = train_mask_datagen.flow_from_directory(train_mask_dir, class_mode=None, seed=seed, color_mode=colorMode, target_size=inputTargetSize, batch_size=batchSize)


#Combine the training image and mask generators to create the training generator to be used
#training_gen = zip(train_image_generator, train_mask_generator)  #zipping for some reason calls the generators for an infinite creation of images (enumerates the list, but infinite generators means no end) - use itertools instead in Python 2.

training_gen = itertools.izip(train_image_generator, train_mask_generator)



#No need to use augmentation parameters for validation; just rescaling needed to define the same kind of generators for the test data set
test_image_datagen = ImageDataGenerator(rescale = 1./255)
test_image_generator = test_image_datagen.flow_from_directory(test_image_dir, class_mode=None, seed=seed, color_mode=colorMode, target_size=inputTargetSize, batch_size=batchSize)

test_mask_datagen = ImageDataGenerator(rescale = 1./255)
test_mask_generator = test_mask_datagen.flow_from_directory(test_mask_dir, class_mode=None, seed=seed, color_mode=colorMode, target_size=inputTargetSize, batch_size=batchSize)

#Combine the testing image and mask generators to create the validation generator - don't use zip because it doesn't work as expected in Python2 (see training_gen definition and comment above)
#validation_gen = zip(test_image_generator, test_mask_generator)

validation_gen = itertools.izip(test_image_generator, test_mask_generator)



##DEBUG FOR VARIOUS PURPOSES
img, msk = next(training_gen)

#plt.imshow(img[0])
#plt.imshow(msk[0])

#plt.imshow(img[1])
#plt.imshow(msk[1])

np.set_printoptions(threshold=np.nan) 

#print img[0].shape
#print img[0].dtype
#print img[0]

#print msk[0].shape
#print msk[0].dtype
#print msk[0]


print("Building Unet of size (" + str(resizeHeight) + ", " + str(resizeWidth) + ")")
print("Loss function is " + str(lossFunc))
print("Learning rate is " + str(learningRate))
print("Number of epochs will be " + str(numEpochs))

raw_input("Press enter to build model with above parameters...")


if not usePrebuiltModel:
    
    if lossFunc=='dice_coef_loss':
        lossFunc = dice_coef
    elif lossFunc=='mean_iou':
        lossFunc = mean_iou
    else:
        lossFunc = 'binary_crossentropy'
        
    
    if resizeHeight==128 and resizeWidth==128:
        model = get_unet_128(input_shape=(resizeHeight, resizeWidth, numChannels), num_classes=1, 
                             learn_rate = learningRate, loss_func=lossFunc, metric_list=[dice_coef])
    elif resizeHeight==256 and resizeWidth==256:
        model = get_unet_256(input_shape=(resizeHeight, resizeWidth, numChannels), num_classes=1, 
                             learn_rate = learningRate, loss_func=lossFunc, metric_list=[dice_coef])
    elif resizeHeight==512 and resizeWidth==512:
        model = get_unet_512(input_shape=(resizeHeight, resizeWidth, numChannels), num_classes=1, 
                             learn_rate = learningRate, loss_func=lossFunc, metric_list=[dice_coef])
    elif resizeHeight==1024 and resizeWidth==1024:
        model = get_unet_1024(input_shape=(resizeHeight, resizeWidth, numChannels), num_classes=1, 
                              learn_rate = learningRate, loss_func=lossFunc, metric_list=[dice_coef])
    else:
        print "Input doesn't match one of the standard UNet sizes, code will error soon because no model was created..."
        print "Yes, I know this is bad practice, and I'll fix it soon to exit gracefully."
        
    model.summary()
    
else:
    model = load_model(loadModelFile, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    model.summary()
    
    
if usePremadeWeights:
    model.load_weights(loadWeightFile)


if doTraining:
    tbOut = TensorBoard(log_dir=saveWeightDir + 'TBlogs/', histogram_freq=1,  
          write_graph=True, write_images=True)
    #earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint(saveWeightDir + CNN_name +'-weights.{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True, save_weights_only=True)

    #results = model.fit_generator(generator=training_gen, epochs=numEpochs, steps_per_epoch=trainEpochSteps, validation_data=validation_gen, validation_steps=valEpochSteps, callbacks=[earlystopper, checkpointer, tbOut])
    
    results = model.fit_generator(generator=training_gen, epochs=numEpochs, steps_per_epoch=trainEpochSteps, validation_data=validation_gen, validation_steps=valEpochSteps, callbacks=[checkpointer, tbOut])
    
    #save the entire model (weights are saved by the ModelCheckpoint)
    model.save(saveModelFile)