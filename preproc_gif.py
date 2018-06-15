import SimpleITK as sitk
import sys, os
from img import *
from PIL import Image

##TODO: Implement batch downsampling/resizing of images in a directory before being passed into the convolutional neural network, so that overall CPU resource usage is decreased and processing speed before training is increased.

#Purpose: to enable the preprocessed images to already exist to be loaded, rather than needing to be constantly preprocessed on the fly, thus requiring only one processing of these images before augmentation and final resizing.

def main():
    
    expectedExt = ".gif"
    outExt = ".png"
    
    raw_image_dir = "raw/images/"
    raw_mask_dir = "raw/masks/"
    dsFactor = 10
    
    out_image_dir = "resized/images/"
    out_mask_dir = "resized/masks/"
    
    if not os.path.exists(out_image_dir): os.makedirs(out_image_dir)
    if not os.path.exists(out_mask_dir): os.makedirs(out_mask_dir)
        
    #preproc(raw_image_dir, out_image_dir, dsFactor, expectedExt, outExt)
    preproc(raw_mask_dir, out_mask_dir, dsFactor, expectedExt, outExt)



def preproc(inImgDir, outImgDir, dsFactor=5, ext=".img", outExt='.png'):
    for img_name in os.listdir(inImgDir):
        if img_name.endswith(ext):
            im = Image.open(inImgDir+img_name)
            
            file_base = os.path.splitext(img_name)[0]
            im.save(outImgDir+file_base+outExt)
            



if __name__ == "__main__": main()