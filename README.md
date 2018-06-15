# UNet-Keras-TF
Basic implementation of the U-Net Convolutional Neural Network Architecture, implemented using Keras with a Tensorflow backend

The dependencies list is not exhaustive - I have only mentioned the packages that came to mind. Others are probably needed, but the versioning is likely less important.

Main dependencies/versions used (for building and running the U-Net):\
numpy >=1.14.4\
scipy >=1.1.0\
tensorflow ==1.2.0 (should work with later versions, but untested)\
tensorflow-gpu ==1.2.0 (can ignore if not processing on GPU, but will be slower)\
Keras ==2.0.6 (should work with later versions, but untested)\
h5py >=2.8.0\
Pillow ==5.1.0


Other dependencies:
SimpleITK =0.9.1 (for preprocessing images only)\ 
matplotlib (only for debug purposes, for the purpose of viewing intermediate images)
