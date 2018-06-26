from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam

import tensorflow as tf
from keras import backend as K

import os


##VERSIONS: Tensorflow 1.2.0; Keras 2.0.6
#Should work with later versions just fine, though some minor changes to imports may be useful.
#The architecture implemented here is a 256x256 U-Net, though I may add support for deeper U-Nets if this works


#SUPER IMPORTANT - DEFINE WHICH GPU TO USE, IF RELEVANT
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # or '1' or whichever GPU is available on your machine
#use nvidia-smi to see which devices are currently in use by one person or another, and use the one not in use.

#Implementation heavily tweaked from https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/blob/master/model/u_net.py to use Conv2DTranspose instead of Upsampling2D+Conv2D; also simplified Conv2D calls



## Define dice coefficient metric and loss function associated with it 
def dice_coef(y_true, y_pred):
    smooth = 1e-5
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



def get_unet_128(input_shape=(128, 128, 1),
                  num_classes=1, learn_rate=1e-4, loss_func='binary_crossentropy', metric_list=[dice_coef]):
    
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    down1 = Conv2D(64, (3, 3), activation='relu', padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down1_pool)
    down2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down2_pool)
    down3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down3_pool)
    down4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), activation='relu', padding='same')(down4_pool)
    center = Conv2D(1024, (3, 3), activation='relu', padding='same')(center)
    # center

    up4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up4)
    up4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up4)
    # 16

    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    # 32

    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    # 64

    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=Adam(lr=learn_rate), loss=loss_func, metrics=[dice_coef])

    return model


def get_unet_256(input_shape=(256, 256, 1),
                  num_classes=1, learn_rate=1e-4, loss_func='binary_crossentropy', metric_list=[dice_coef]):
    
    inputs = Input(shape=input_shape)
    # 256

    down0 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    down0 = Conv2D(32, (3, 3), activation='relu', padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), activation='relu', padding='same')(down0_pool)
    down1 = Conv2D(64, (3, 3), activation='relu', padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down1_pool)
    down2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down2_pool)
    down3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down3_pool)
    down4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), activation='relu', padding='same')(down4_pool)
    center = Conv2D(1024, (3, 3), activation='relu', padding='same')(center)
    # center

    up4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up4)
    up4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up4)
    # 16

    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    # 32

    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    # 64

    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    # 128

    up0 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), activation='relu', padding='same')(up0)
    up0 = Conv2D(32, (3, 3), activation='relu', padding='same')(up0)
    # 256

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=Adam(lr=learn_rate), loss=loss_func, metrics=[dice_coef])

    return model



def get_unet_512(input_shape=(512, 512, 1),
                  num_classes=1, learn_rate=1e-4, loss_func='binary_crossentropy', metric_list=[dice_coef]):
    
    inputs = Input(shape=input_shape)
    # 512

    down0a = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    down0a = Conv2D(16, (3, 3), activation='relu', padding='same')(down0a)
    down0a_pool = MaxPooling2D((2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), activation='relu', padding='same')(down0a_pool)
    down0 = Conv2D(32, (3, 3), activation='relu', padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), activation='relu', padding='same')(down0_pool)
    down1 = Conv2D(64, (3, 3), activation='relu', padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down1_pool)
    down2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down2_pool)
    down3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down3_pool)
    down4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), activation='relu', padding='same')(down4_pool)
    center = Conv2D(1024, (3, 3), activation='relu', padding='same')(center)
    # center

    up4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up4)
    up4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up4)
    # 16

    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    # 32

    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    # 64

    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    # 128

    up0 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), activation='relu', padding='same')(up0)
    up0 = Conv2D(32, (3, 3), activation='relu', padding='same')(up0)
    # 256

    up0a = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), activation='relu', padding='same')(up0a)
    up0a = Conv2D(16, (3, 3), activation='relu', padding='same')(up0a)
    # 512

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0a)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=Adam(lr=learn_rate), loss=loss_func, metrics=[dice_coef])

    return model



def get_unet_1024(input_shape=(1024, 1024, 1),
                  num_classes=1, learn_rate=1e-4, loss_func='binary_crossentropy', metric_list=[dice_coef]):
    
    inputs = Input(shape=input_shape)
    # 1024

    down0b = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    down0b = Conv2D(8, (3, 3), activation='relu', padding='same')(down0b)
    down0b_pool = MaxPooling2D((2, 2))(down0b)
    # 512

    down0a = Conv2D(16, (3, 3), activation='relu', padding='same')(down0b_pool)
    down0a = Conv2D(16, (3, 3), activation='relu', padding='same')(down0a)
    down0a_pool = MaxPooling2D((2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), activation='relu', padding='same')(down0a_pool)
    down0 = Conv2D(32, (3, 3), activation='relu', padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), activation='relu', padding='same')(down0_pool)
    down1 = Conv2D(64, (3, 3), activation='relu', padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down1_pool)
    down2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down2_pool)
    down3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down3_pool)
    down4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), activation='relu', padding='same')(down4_pool)
    center = Conv2D(1024, (3, 3), activation='relu', padding='same')(center)
    # center

    up4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up4)
    up4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up4)
    # 16

    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    # 32

    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    # 64

    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    # 128

    up0 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), activation='relu', padding='same')(up0)
    up0 = Conv2D(32, (3, 3), activation='relu', padding='same')(up0)
    # 256

    up0a = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), activation='relu', padding='same')(up0a)
    up0a = Conv2D(16, (3, 3), activation='relu', padding='same')(up0a)
    # 512

    up0b = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(8, (3, 3), activation='relu', padding='same')(up0b)
    up0b = Conv2D(8, (3, 3), activation='relu', padding='same')(up0b)
    # 1024

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=Adam(lr=learn_rate), loss=loss_func, metrics=[dice_coef])

    return model