import SimpleITK as sitk
import sys, os
from img import *


##TODO: Implement batch downsampling/resizing of images in a directory before being passed into the convolutional neural network, so that overall CPU resource usage is decreased and processing speed before training is increased.

#Purpose: to enable the preprocessed images to already exist to be loaded, rather than needing to be constantly preprocessed on the fly, thus requiring only one processing of these images before augmentation and final resizing.

def main():
    
    expectedExt = ".img"
    outExt = ".png"
    
    raw_image_dir = "raw/images/"
    raw_mask_dir = "raw/masks/"
    dsFactor = 10
    
    out_mask_dir = "resized/masksInv/"
    
    if not os.path.exists(out_mask_dir): os.makedirs(out_mask_dir)
        
    preproc(raw_image_dir, raw_mask_dir, out_mask_dir, dsFactor/10, expectedExt, outExt)



def preproc(inImgDir, inMaskDir, outMaskDir, dsFactor=5, ext=".img", outExt='.png'):
    for img_name in os.listdir(inImgDir):
        if img_name.endswith(ext):            
            file_base = os.path.splitext(img_name)[0]
            img_orig = imgRead(inImgDir + img_name)
            mask_orig = imgRead(inMaskDir + file_base+'_GM.img')
            
            if not os.path.exists(outMaskDir + 'bgmask/'): os.makedirs(outMaskDir + 'bgmask/')
            if not os.path.exists(outMaskDir + 'depths/'): os.makedirs(outMaskDir + 'depths/')
            if not os.path.exists(outMaskDir + 'WM_I/'): os.makedirs(outMaskDir + 'WM_I/')
            if not os.path.exists(outMaskDir + 'I_only/'): os.makedirs(outMaskDir + 'I_only/')

            
            img_orig = sitk.RescaleIntensity(img_orig, 0, 255)
            
            bgMask_orig = sitk.BinaryThreshold(img_orig, 220, 255)
            brainMask_orig = 1 - bgMask_orig
            bgMask_orig = imgLargestMaskObject(1-imgLargestMaskObject(brainMask_orig)) #take largest continuous region for the brain, and then use that to generate mask for the background and take largest region of that for overall background mask
            brainMask_orig = 1 - bgMask_orig
            imgWrite(bgMask_orig, outMaskDir + 'bgmask/' + file_base+'_bgmask.img')
                
            #We will compute the distance map (depth image) for later use
            depth_map = sitk.DanielssonDistanceMap(bgMask_orig, inputIsBinary=True, useImageSpacing=True)
            imgWrite(depth_map, outMaskDir + 'depths/' + file_base+'_depth.img')
            
            resHeight = np.array(brainMask_orig.GetSpacing())[0];
            maskHeight = np.array(mask_orig.GetSpacing())[0];
            
            scaleFactor = float(maskHeight)/float(resHeight)
            print scaleFactor
            
            newSpacing = scaleFactor * np.array(brainMask_orig.GetSpacing())
            
            print newSpacing
            
            brainMask_orig = imgResample(brainMask_orig, spacing=newSpacing)
            depth_map = imgResample(depth_map, spacing=newSpacing)
            
            print mask_orig.GetOrigin()   #debug
            print brainMask_orig.GetOrigin()   #debug
            
            mask_orig.SetOrigin((0,0))
            
            layer_WM_masks = sitk.Mask(1-mask_orig, brainMask_orig)
            imgWrite(layer_WM_masks, outMaskDir + 'WM_I/' + file_base+'_plusWM.img')
            
            print layer_WM_masks.GetSize() 
            
            layerI_depth = depth_map < 0.5
            print depth_map.GetOrigin()   #debug
            print depth_map.GetSize()
            
            
            layer_I_mask = sitk.And(layer_WM_masks == 1, layerI_depth)
            imgWrite(layer_I_mask, outMaskDir + 'I_only/' + file_base+'_I.img')
            
            img_out = imgLargestMaskObject(layer_I_mask)
            img_out = sitk.Cast(sitk.RescaleIntensity(img_out), sitk.sitkUInt8) 
            imgWrite(img_out, outMaskDir + file_base + "_I" + outExt)
            
            
            #if dsFactor > 1:
                #origSpacing = np.array(img_orig.GetSpacing())
                #dsSpacing = dsFactor*origSpacing
                #img_out = imgResample(img_orig, spacing=dsSpacing)
            #else:
                #img_out = img_orig
                #img_out = sitk.Cast(sitk.RescaleIntensity(img_out), sitk.sitkUInt8) 
            #file_base = os.path.splitext(img_name)[0]
            #imgWrite(img_out, outImgDir + file_base + "_ds" + outExt)
            



if __name__ == "__main__": main()