#!/usr/bin/env python
import SimpleITK as sitk
import numpy as np
import sys, itertools
import math
sitkToNpDataTypes = {sitk.sitkUInt8: np.uint8,
                     sitk.sitkUInt16: np.uint16,
                     sitk.sitkUInt32: np.uint32,
                     sitk.sitkInt8: np.int8,
                     sitk.sitkInt16: np.int16,
                     sitk.sitkInt32: np.int32,
                     sitk.sitkFloat32: np.float32,
                     sitk.sitkFloat64: np.float64,
                     }

def imgCollaspeDimension(inImg):
    inSize = inImg.GetSize()
    inDimension = inImg.GetDimension()
    if inSize[inDimension-1] == 1:
        outSize = list(inSize)
        outSize[inDimension-1] = 0
        outIndex = [0]*inDimension
        inImg = sitk.Extract(inImg, outSize, outIndex, 1)
    return inImg

def imgRead(path):
    inImg = sitk.ReadImage(path)
    inImg = imgCollaspeDimension(inImg)
    return inImg

    
def imgWrite(img, path):
    sitk.WriteImage(img, path)
    
def imgResample(img, spacing, size=[], useNearest=False, origin=[], outsideValue=0):
    """
    Resamples image to given spacing and size.
    """
    if len(spacing) != img.GetDimension(): raise Exception("len(spacing) != " + str(img.GetDimension()))

    # Set Size
    if size == []:
        inSpacing = img.GetSpacing()
        inSize = img.GetSize()
        size = [int(math.ceil(inSize[i]*(inSpacing[i]/spacing[i]))) for i in range(img.GetDimension())]
    else:
        if len(size) != img.GetDimension(): raise Exception("len(size) != " + str(img.GetDimension()))
    
    if origin == []:
        origin = [0]*img.GetDimension()
    else:
        if len(origin) != img.GetDimension(): raise Exception("len(origin) != " + str(img.GetDimension()))
    
    # Resample input image
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
    identityTransform = sitk.Transform()
    identityDirection = list(sitk.AffineTransform(img.GetDimension()).GetMatrix())

    return sitk.Resample(img, size, identityTransform, interpolator, origin, spacing, identityDirection, outsideValue)

def imgBC(img, mask=None, scale=1.0, numBins=64, returnBias=False):
    """
    Bias corrects an image using the N4 algorithm
    """
    spacing = np.array(img.GetSpacing())/scale
    img_ds = imgResample(img, spacing=spacing)

    # Calculate bias
    if mask is None:
        mask_ds = sitk.Image(img_ds.GetSize(), sitk.sitkUInt8)+1
        mask_ds.CopyInformation(img_ds)
    else:
        mask_ds = imgResample(mask, spacing=spacing, useNearest=True)
        mask_ds = mask_ds > 0

    splineOrder = 2
    img_ds_bc = sitk.N4BiasFieldCorrection(sitk.Cast(img_ds, sitk.sitkFloat32), mask_ds, numberOfHistogramBins=numBins, splineOrder=splineOrder, numberOfControlPoints=[splineOrder+1]*4)
    #bias_ds = img_ds_bc - sitk.Cast(img_ds,img_ds_bc.GetPixelID())

    bias_ds = imgFinite(img_ds_bc / img_ds)
    bias_ds = sitk.Mask(bias_ds, mask_ds) + sitk.Cast(1-mask_ds, sitk.sitkFloat32) # Fill background with 1s

    # Upsample bias    
    bias = imgResample(bias_ds, spacing=img.GetSpacing(), size=img.GetSize())
    bias = sitk.Cast(bias, img.GetPixelID())

    # Apply bias to original image and threshold to eliminate negitive values
    try:
        upper = np.iinfo(sitkToNpDataTypes[img.GetPixelID()]).max
    except:
        upper = np.finfo(sitkToNpDataTypes[img.GetPixelID()]).max *0.99

    #img_bc = sitk.Threshold(img + sitk.Cast(bias, img.GetPixelID()),
    #                        lower=0,
    #                        upper=upper)

    img_bc = sitk.Threshold(img * bias, lower=0, upper=upper)

    if returnBias:
        return (img_bc, bias)
    else:
        return img_bc


def imgPreprocess(inImg, spacing):
    sigma = np.mean(spacing)

    #img = sitk.DiscreteGaussian(inImg, 2*sigma**2)
    img = sitk.Median(inImg, [20,20])
    img = sitk.SmoothingRecursiveGaussian(img, sigma)
    #imgWrite(img, "/cis/home/kwame/Projects/akm/dat/out.img")
    #sys.exit()
    
    img =  imgResample(img, spacing)
    mask = sitk.BinaryThreshold(img, 0,170)
    img = sitk.Threshold(img, 0, 170)

    #stats = sitk.StatisticsImageFilter()
    #stats.Execute(inImg)
    #img = stats.GetMaximum() - img

    #sitk.WriteImage(img, "/cis/home/kwame/Projects/akm/dat/before.img")
    img = imgBC(img, numBins=64, scale=0.5)
    #sitk.WriteImage(img, "/cis/home/kwame/Projects/akm/dat/after.img")
    
    return (img, mask)

def imgProcessNew2(inImg, distImg):
    depthStep = 0.1
    depthList = np.arange(0,2, depthStep)
    epsilon = 1e-6
    outList = []
    for i in range(len(depthList)):
        lower = depthList[i]
        if i == len(depthList)-1:
            upper = np.inf
        else:
            upper = depthList[i+1] - epsilon
            
        depthMask = sitk.Cast(sitk.BinaryThreshold(distImg, lowerThreshold=lower, upperThreshold=upper), sitk.sitkUInt8)
        inMasked = sitk.Mask(255-inImg, depthMask)
        #sigma = np.mean(inImg.GetSpacing())
        #inSmoothed = sitk.DiscreteGaussian(inMasked, 2*sigma**2)
        inSmoothed = sitk.GrayscaleMorphologicalClosing(inMasked, 20)
        #inSmoothed = sitk.GrayscaleDilate(inMasked,10)
        inSmoothed = sitk.Mask(inSmoothed, depthMask)

        
        #if i == 1:
        #    imgWrite(inSmoothed, "/cis/home/kwame/Projects/akm/dat/out.img")
        outList.append(inSmoothed)

    outImg = sitk.NaryAdd(outList)
    #outImg = imgBC(img, numBins=16, scale=0.25)
    imgWrite(255-outImg, "cis/home/rkumar/Desktop/akmtest/dat/out.img")

    sys.exit()


def imgPreprocessNew(inImg, spacing):
    sigma = np.mean(spacing)

    
    
    #img = sitk.SmoothingRecursiveGaussian(inImg, sigma)
    #imgWrite(img, "/cis/home/kwame/Projects/akm/dat/out.img")
    #sys.exit()
    
 
    img = sitk.Cast(inImg, sitk.sitkUInt8)
    img = sitk.Median(img, [20,20])

    
    img = sitk.GrayscaleMorphologicalOpeninlg(img,5)

    """
    img = sitk.Cast(inImg, sitk.sitkFloat32)
    sigma = np.mean(spacing)
    img = sitk.DiscreteGaussian(img, sigma**2)
    """
    img =  imgResample(img, spacing)
    
    
    mask = sitk.BinaryThreshold(img, 0,170)
    img = sitk.Threshold(img, 0, 170)




    #img = imgBC(img, numBins=16, scale=0.5)

    
    return (img, mask)

"""
def imgPreprocess(inImg, spacing):
    sigma = np.mean(spacing)
    img = sitk.DiscreteGaussian(inImg, sigma**2)
    img =  imgResample(img, spacing)
    img = sitk.Median(img, [2,2])
    mask = sitk.BinaryThreshold(img, 0,170)
    img = sitk.Threshold(img, 0, 170)

    #stats = sitk.StatisticsImageFilter()
    #stats.Execute(inImg)
    #img = stats.GetMaximum() - img

    #sitk.WriteImage(img, "/cis/home/kwame/Projects/akm/dat/before.img")
    img = imgBC(img, numBins=64, scale=1)
    #sitk.WriteImage(img, "/cis/home/kwame/Projects/akm/dat/after.img")
    
    return (img, mask)

"""
def imgNormalize(img):
    """
    Normalize image to unity sum
    """    
    constant = np.sum(sitk.GetArrayFromImage(img))*np.prod(img.GetSpacing())
    return img/constant

def imgHist(imgList, binsList):
    dataList = []
    for img in imgList[::-1]: dataList.append(sitk.GetArrayFromImage(img).flatten())
    data = np.vstack(dataList).T
    hist, edges = np.histogramdd(data, bins=binsList[::-1])

    histSpacing = []
    for i in range(len(edges))[::-1]: histSpacing.append(float(edges[i][1]-edges[i][0]))
    #hist /= np.sum(hist)*np.prod(histSpacing)  # Normalize by sum and spacing

    histImg = sitk.GetImageFromArray(hist)
    histImg.SetSpacing(histSpacing)
    
    return imgNormalize(histImg)

def imgFinite(inImg):
    """
    Zeros out non finite values (Infs and NaNs)
    """
    arr = sitk.GetArrayFromImage(inImg)
    arr[~np.isfinite(arr)]=0
    outImg = sitk.GetImageFromArray(arr)
    outImg.SetSpacing(inImg.GetSpacing())
    outImg.SetOrigin(inImg.GetOrigin())
    outImg.SetDirection(inImg.GetDirection())
    return outImg


def imgPoints(size, spacing=None, origin=None):
    numDims = len(size)
    if spacing is None: spacing = [1]*numDims
    if origin is None: origin = [0]*numDims
    if len(size) != len(spacing): raise Exception("len(size) != len(spacing)")

    coordsImg = sitk.PhysicalPointSource(size=size, spacing=spacing, origin=origin)
    coordsArray = sitk.GetArrayFromImage(coordsImg)
    coords = []
    for i in range(numDims): coords.append(sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(coordsImg,i)).flatten())
    points = np.vstack(coords).T
    return points

def imgHistStats(histImg):
    hist = sitk.GetArrayFromImage(histImg)
    histSpacing = np.array(histImg.GetSpacing())[::-1]
    numDims = histImg.GetDimension()
    # Get coordinates
    #coords2 = GetCoords(histImg.GetSize(), histImg.GetSpacing())
    points = imgPoints(histImg.GetSize(), histImg.GetSpacing(), histImg.GetOrigin())

    coords = []
    for i in range(numDims): coords.append(points[:,i].reshape(hist.shape))
    
    weight = np.sum(hist)*np.prod(histSpacing)

    
    # Calculate mu
    mu = np.zeros(numDims)
    for i in range(numDims): mu[i] = np.sum(coords[i]*hist)*np.prod(histSpacing)
    mu /= weight
    
    # Calculate covariance
    V = np.zeros((numDims, numDims))
    dimPairList = map(np.array,itertools.combinations_with_replacement(range(numDims),2))
    """
    for dimPair in dimPairList:
        histCoordProd = hist.copy()
        for i in dimPair: histCoordProd *= coords[i]
        V[dimPair[0],dimPair[1]] = np.sum(histCoordProd)*np.prod(histSpacing) - np.prod(mu[dimPair])
    """
    for dimPair in dimPairList:
        V[dimPair[0],dimPair[1]] = np.sum((coords[dimPair[0]] - mu[dimPair[0]])*(coords[dimPair[1]] - mu[dimPair[1]])*hist*np.prod(histSpacing) / weight)
        
        #histCoordProd = hist.copy()
        #for i in dimPair: histCoordProd *= coords[i]
        #V[dimPair[0],dimPair[1]] = np.sum(histCoordProd)*np.prod(histSpacing) - np.prod(mu[dimPair])

        
    V = V + V.T - np.diag(V.diagonal()) # Symetrize V
    return (weight, mu,V)

def imgLargestMaskObject(maskImg):
    ccFilter = sitk.ConnectedComponentImageFilter()
    labelImg = ccFilter.Execute(maskImg)
    numberOfLabels = ccFilter.GetObjectCount()
    labelArray = sitk.GetArrayFromImage(labelImg)
    labelSizes = np.bincount(labelArray.flatten())
    largestLabel = np.argmax(labelSizes[1:])+1
    outImg = sitk.GetImageFromArray((labelArray==largestLabel).astype(np.int16))
    outImg.CopyInformation(maskImg) # output image should have same metadata as input mask image
    return outImg
    
#def imgLargestMaskObject(maskImg, bgSeed=(1,1,1), numIter=1, mult=2.5, initNeighbor=1):
    #outImg = sitk.ConfidenceConnected(maskImg, seedList=[bgSeed],numberOfIterations=numIter,multiplier=mult,initialNeighborhoodRadius=initNeighbor,replaceValue=1)
    #outImg.CopyInformation(maskImg) # output image should have same metadata as input mask image
    #return outImg
