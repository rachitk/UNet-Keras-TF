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
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

import tensorflow as tf

from scipy.misc import imresize
from tqdm import tqdm

#others imported sklearn, but this is unneeded with the implementation of keras' image.flow_from_directory

import os

import itertools

from PIL import Image

import SimpleITK as sitk

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # or '1' or whichever GPU is available on your machine

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

## Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def load_image(infilename):
    img = Image.open(infilename).convert('L')
    img.load()
    img = img.resize((512,512), Image.ANTIALIAS)
    data = np.asarray(img, dtype="float32")
    return data

def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata,0,255), dtype="uint8"), "L")
    img.save(outfilename)



loadModelFile = 'modelFiles/fullModel.h5'   #Define where the model should be loaded from if using a prebuilt one

inFile = 'input/try/0.png'


model = load_model(loadModelFile, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef, 
                                                  'mean_iou': mean_iou})

im = load_image(inFile)
save_image(im, 'input/try/0_res.png')

im = im * 1./255

im = np.reshape(im, (1,512,512,1))

out = model.predict(im)

outFile = 'input/try_out/0_pred.img'

out = np.reshape((out), (512,512))
out = out * (255.0 / out.max())

#np.set_printoptions(threshold=np.nan)   #debug only
#print out.dtype
#print out

outImg = sitk.GetImageFromArray(out)
sitk.WriteImage(outImg, outFile)

#save_image(out, outFile)
