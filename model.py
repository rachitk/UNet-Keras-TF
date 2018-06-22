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
resizeHeight = 256  #height of image to be entered into CNN
resizeWidth = 256  #width of image to be entered into CNN
numChannels = 1   #number of channels of image to be entered into CNN; 1 is greyscale, 3 is rgb  --should always be set to 1, because the CNN can't handle 3 as it is currently.

batchSize = 8   #Define batch sizes (how many images will be loaded into memory at a time) - same for both training and validation in this implementation

trainEpochSteps = 64  #number of training steps per epoch (how many batches to take out per epoch), typically the number of training images you have divided by the number of images in each batch to cover all the images
valEpochSteps = 32  #number of validation steps per epoch (how many batches to take out per epoch), typically the number of validation images you have divided by the number of images in each batch to cover all the images

learningRate = 1e-4  #this is almost always this value for the Adam optimizer, though other optimizers may not even use a learning rate value or may use different ones.

numEpochs = 200  #one of the most important things: how many epochs should the network go through before ending?



#Define values for the run
usePrebuiltModel = False   #Don't build the U-Net model from scratch (do this if you only have the weights, otherwise set this to false to build the model from scratch

doTraining = True   #Don't train the model. This will cause errors or strange behavior if you don't preload weights from somewhere.

usePremadeWeights = False   #If using premade weights, then load them instead of the model's weights or instead of training

loadModelFile = 'modelFiles/newFullModel.h5'   #Define where the model should be loaded from if using a prebuilt one

saveModelFile = 'modelFiles/fullModel.h5'   #Define where the model should be saved if doing training

loadWeightFile = 'modelFiles/weights/fullWeights.h5'    #Define where to load the weight file from if using premade weights

saveWeightDir = 'modelFiles/weights/'    #Define where to save the progressive weights generated if training




#Define data augmentation parameters for training
data_aug_param = dict(rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     rescale=1./255,
                     shear_range = 0.2,
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



#Define dice coefficient function as metric
#def dice_coeff(y_true, y_pred):
    #smooth = 1e-5
    
    #y_true = tf.round(tf.reshape(y_true, [-1]))
    #y_pred = tf.round(tf.reshape(y_pred, [-1]))
    
    #isct = tf.reduce_sum(y_true * y_pred)
    
    #return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    
def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
    
## Define IoU metric
#def mean_iou(y_true, y_pred):
    #prec = []
    #for t in np.arange(0.5, 1.0, 0.05):
        #y_pred_ = tf.to_int32(y_pred > t)
        #score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        #K.get_session().run(tf.local_variables_initializer())
        #with tf.control_dependencies([up_opt]):
            #score = tf.identity(score)
        #prec.append(score)
    #return K.mean(K.stack(prec), axis=0)

    

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
#img, msk = next(training_gen)

#plt.imshow(img[0])
#plt.imshow(msk[0])

#plt.imshow(img[1])
#plt.imshow(msk[1])

#print img[0].shape
#print img[0].dtype

#print msk[0].shape
#print msk[0].dtype

#raw_input("Press Enter to continue...")

if not usePrebuiltModel:
    #U-Net architecture implementation

    inputs = Input((resizeHeight, resizeWidth, numChannels))

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learningRate), loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=Adam(learningRate), loss='binary_crossentropy', metrics=[dice_coef])
    model.summary()
else:
    model = load_model(loadModelFile, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    
    
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