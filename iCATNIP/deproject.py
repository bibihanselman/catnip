import numpy as np
import cv2
import math

def deproject(data, imdat, PA, incl):
    '''
    Deprojects the image data using disk PA and incl.
    '''
    ndimx,ndimy = imdat['Dimensions'][1],imdat['Dimensions'][0]
    xcen,ycen = imdat['Center'][1], imdat['Center'][0]

    #rotate image by PA
    M = cv2.getRotationMatrix2D((xcen,ycen), PA - 90, 1.0)

    imrot = cv2.warpAffine(data, M,
                           (ndimx, ndimy),
                           flags = cv2.INTER_CUBIC)

    #stretch image in y by cos(incl) to deproject and divide by that factor to preserve flux
    im_rebin = cv2.resize(imrot,
                          (ndimx,int(np.round(ndimy*(1/math.cos(math.radians(incl)))))),
                          interpolation = cv2.INTER_CUBIC)
    im_rebin = im_rebin/(1./math.cos(math.radians(incl)))

    ndimy2 = im_rebin.shape[0]
    ycen2 = int(round((ndimy2/ndimy)*ycen))
    
    #rotate back to original orientation
    M = cv2.getRotationMatrix2D((xcen, ycen2), -1*(PA - 90), 1.0)

    im_rebin_rot = cv2.warpAffine(im_rebin, M,
                                 (ndimx, ndimy2),
                                 flags = cv2.INTER_CUBIC)

    return im_rebin_rot