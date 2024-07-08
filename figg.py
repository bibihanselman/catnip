"""
FIGG PROCESSING:
normalizes, background subtracts, and scales disk images
according to user-defined parameters.
---
Original code written by Alex DelFranco
Adapted by Bibi Hanselman
Original dated 7 July 2024
Updated 8 July 2024
"""
import numpy as np
#import pandas as pd
#from pylab import figure, cm
#import matplotlib
#from matplotlib.colors import LogNorm
#import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.visualization import (AsinhStretch, ImageNormalize)
#import glob
#import astropy.units as u
#import math
from astropy.visualization.interval import ManualInterval
#from astropy.visualization import ZScaleInterval
#import astropy.convolution as conv
#import os
from astropy.io import fits
#from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
#from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve
import astropy.stats as stats
import scipy.stats

###################### ASINHLIN SCALING ######################

from astropy.visualization.stretch import BaseStretch

######################

def _prepare(values, clip=True, out=None):
    """
    Prepare the data by optionally clipping and copying, and return the
    array that should be subsequently used for in-place calculations.
    """

    if clip:
        return np.clip(values, 0., 1., out=out)
    else:
        if out is None:
            return np.array(values, copy=True)
        else:
            out[:] = np.asarray(values)
            return out

######################

class AsinhLinStretch(BaseStretch):
    """
    A modified asinh stretch.

    The stretch is given in part by:

    .. math::
        y = \frac{{\rm asinh}(x / a)}{{\rm asinh}(1 / a)}.

    Parameters
    ----------
    a : float, optional
        The ``a`` parameter used in the above formula.  The value of
        this parameter is where the asinh curve transitions from linear
        to logarithmic behavior, expressed as a fraction of the
        normalized image.  ``a`` must be greater than 0 and less than or
        equal to 1 (0 < a <= 1).  Default is 0.1.
    """

    def __init__(self, a=0.1, b=0.5, c=0.7):
        super().__init__()
        if a <= 0 or a > 1:
            raise ValueError("a must be > 0 and <= 1")
        self.a = a
        if b <= 0 or b > 1:
            raise ValueError('b must be > 0 and <= 1')
        self.b = b
        if c <= b or c > 1:
            raise ValueError('c must be > b and <= 1')
        self.c = c

    def __call__(self, values, clip=True, out=None):
        raw = _prepare(values, clip=clip, out=out)
        # Calculate transition back to linear
        n = np.arcsinh(self.b / self.a) / np.arcsinh(1. / self.a)
        # Define ranges
        r1,r2,r3 = raw.copy(),raw.copy(),raw.copy()
        r1[r1 > self.b] = 0
        r2[r2 <= self.b] = 0
        r2[r2 > self.c] = 0
        r3[r3 <= self.c] = 0
        # Calculate range 1
        np.true_divide(r1, self.a, out=r1)
        np.arcsinh(r1, out=r1)
        np.true_divide(r1, np.arcsinh(1. / self.a), out=r1)
        # Calculate range 2
        if len(r2) > 0:
          r2base = np.multiply((r2),((n-self.c)/(self.b-self.c)))
          r2[r2>0] = self.c * ((n-self.b)/(self.b-self.c))
          r2 = r2base - r2
        values = r1+r2+r3
        return values

###################### IMAGE PROCESSING ######################

old_prc_ims = {}
old_prc_scales = {}
old_imdat = {}

def edge_remove(im,imdat):
  """
  Sets pixel values along the edge of an image to nan.
  The thickness of the edges is determined by an adjustable parameter 'Crop',
  to be defined in imdat.
  
  Inputs:
      im (np array): fits file
      imdat (dict): dictionary containing processing parameters of the desired image
  Outputs:
      im (np array): input file with removed edges
  """
  # Defines the radius of pixels to remove from the image edge
  # (reduced by a percentage so there is overlap)
  edge_crop = int((imdat['Dimensions'] - imdat['Crop'])/2 * 0.95)

  # Crop the image by setting all extra values to nans
  im[-1*edge_crop:] = np.nan
  im[:edge_crop] = np.nan
  im[:,-1*edge_crop:] = np.nan
  im[:,:edge_crop] = np.nan

  # Return the cropped image
  return im

######################

def mask_remove(im,imdat):
  """
  Sets pixel values within a given radius from the image center to nan.
  The radius is determined by an adjustable parameter 'Radius', to be
  defined in imdat.
  
  Inputs:
      im (np array): fits file
      imdat (dict): dictionary containing processing parameters of the desired image
  Outputs:
      im (np array): input file with removed mask
  """
  # Defines the radius of pixels to remove from the image center
  # (reduced by a percentage so there is overlap)
  r = imdat['Radius'] * 0.85

  # Loop through each pixel in the image
  for i in range(-1*int(len(im)/2),int(len(im)/2)):
    for j in range(-1*int(len(im)/2),int(len(im)/2)):
      # If it is inside the radius of the mask
      if (i**2 + j**2 < r**2):
        # Set pixel value to nan
        im[i + int(len(im)/2)][j + int(len(im)/2)] = np.nan

  # Return the new image
  return im

######################

def norm(imdat):
  """
  Takes a .fits file path and returns a normalized and trimmed fits data file
  with an empty header.
  Inputs: 
      imdat (dict): dictionary of image processing parameters for the desired image
  Output: 
      norm_im (np array): normalized fits object with header that has the extreme values cut
  """
  
  # Get fits data
  im = fits.getdata(imdat['Path'])

  # Choose a corresponding cube slice
  if len(im.shape)!=2:
    if imdat['Cube Slice'] != '': im = im[imdat['Cube Slice']]

  # Remove edges and data behind the mask
  if imdat['Crop'] != '':
    im = edge_remove(im,imdat)
  if imdat['Masks'] in ('TRUE', True):
    im = mask_remove(im,imdat)

  if imdat['Mode Subtract'] in ('TRUE', True):
    mode = scipy.stats.mode(im,axis=None,nan_policy='omit')[0]
    im = im - mode
    im[im < 0] = np.nan

  # Normalize the image
  if imdat['Normalization'] in ('TRUE', True):
    bounds = imdat['N-Range'].split(':')
    # Scale lower percentile of the image to 0
    im -= np.nanpercentile(im,float(bounds[0]))
    # Normalize the image to the upper percentile
    norm_im = im/np.nanpercentile(im,float(bounds[1]))

  # Find the median of the background by sigma clipping
  if imdat['Clipping'] in ('TRUE', True):
    bkg = stats.sigma_clip(norm_im,imdat['σ'])
    norm_im = norm_im - np.ma.median(bkg)

  # Check if we want to smooth
  if imdat['Smoothing'] in ('TRUE', True):
    # We smooth with a Gaussian kernel
    kernel = Gaussian2DKernel(x_stddev=float(imdat['Stdev']))
    norm_im = convolve(norm_im, kernel, nan_treatment='interpolate')
    
  # Set colorbar bounds for scaling 
  if imdat['Colorbar'] in ('TRUE', True):
    cb = imdat['CB-Range'].split(':')
    cb = [float(cb[0]),float(cb[1])]
  else: cb = [0,1]

  # Log scale the image data (Hyperbolic Arcsine)
  if imdat['Scaling'] in ('TRUE', True):
    if imdat['Curve'] == 'asinh':
      normalize = ImageNormalize(norm_im, interval=ManualInterval(cb[0],cb[1]), stretch=AsinhStretch(imdat['Scale Parameter']))
    if imdat['Curve'] == 'asinhlin':
      normalize = ImageNormalize(norm_im, interval=ManualInterval(cb[0],cb[1]),
                                stretch=AsinhLinStretch(a=imdat['Scale Parameter'],
                                b=imdat['Scale Parameter 2'],c=imdat['Scale Parameter 3']))    
  else: normalize = 'linear'
  
  # Return the final processed image and its scaling
  return norm_im, normalize

######################

def check_keys(imdt, im):
    """
    Checks if required keys in image dictionary are present for processing.
    If there are any missing keys, raises an error listing them.
    """
    # First, check the required keys
    imdat = imdt[im]
    keys = [
        'Path', 'Cube Slice', 'Crop', 'Dimensions', 'Masks', 'Mode Subtract',
        'Normalization', 'Clipping', 'Smoothing', 'Colorbar', 'Scaling'
        ]
    missingkeys = [key for key in keys if key not in imdat]
    
    # Now check for the 'optional' keys
    # Not an elegant way to do this. Will figure out something better later. 7/8/24
    optkeys = {
        'Masks': 'Radius',
        'Normalization': 'N-Range',
        'Clipping': 'σ',
        'Smoothing': 'Stdev',
        'Colorbar': 'CB-Range'
        }
    for reqkey in optkeys:
        if reqkey in imdat:
            if (imdat[reqkey] in ('TRUE', True)) and (optkeys[reqkey] not in imdat):
                missingkeys.append(optkeys[reqkey])
    #The special case of scaling parameters
    if 'Scaling' in imdat:
        if (imdat['Scaling'] in ('TRUE', True)):
            if 'Curve' not in imdat:
                missingkeys.append('Curve')
            elif imdat['Curve'] in ('asinh', 'asinhlin'):
                if 'Scale Parameter' not in imdat:
                    missingkeys.append('Scale Parameter')
                if imdat['Curve'] == 'asinhlin':
                    if 'Scale Parameter 2' not in imdat:
                        missingkeys.append('Scale Parameter 2')
                    if 'Scale Parameter 3' not in imdat:
                        missingkeys.append('Scale Parameter 3')

    
    if missingkeys != []:
        raise KeyError('Your dictionary for ' + im + ' is missing the following keys: ' 
                       + str(missingkeys) + 
                       '. Please refer to the figg.process() docstring for more info.')

######################

def process(imdat):
  """
  Processes a set of images with normalization, background subtracting, 
  convolutional smoothing, etc. based on given parameters for each image.
  
  Inputs:
      imdat (dict): dictionary in which each key must be the name of the disk 
      whose image is to be processed, and the value of each key a 
      nested dictionary describing the processing settings for each image 
      that MUST contain the following keys (with possible values). 
      Keys marked with * are only required if another key is valued
      'TRUE' or True.
          'Path' (str): name of the fits file
          'Crop' (int): radius to be cropped from the edge of the image
          'Dimensions' (int): length of one side of the image in pixels
          'Masks' (bool; str, 'TRUE' or 'FALSE'): remove pixels within some 
              radius from the center? 
          *'Radius' (float): radius from the center below which pixels are to 
              be removed
          'Mode Subtract' (bool; str, 'TRUE' or 'FALSE'): subtract the mode
              brightness from each pixel value?
          'Normalization' (bool; str, 'TRUE' or 'FALSE'): normalize the 
              pixel brightnesses?
          *'N-Range' (str, formatted: lower:upper): percentile bounds of 
              the normalization
          'Clipping' (bool; str, 'TRUE' or 'FALSE'): perform sigma clipping to
              shave extreme pixel values?
          *'σ' (float): sigma value at which to clip
          'Smoothing' (bool; str, 'TRUE' or 'FALSE'): perform convolutional 
              smoothing on the image?
          *'Stdev' (float): standard deviation of smoothing kernel
          'Colorbar' (bool; str, 'TRUE' or 'FALSE'): set custom colorbar bounds?
          *'CB-Range' (str, formatted: lower:upper): lower and upper bounds
              of colorbar between 0 and 1
          'Scaling' (bool; str, 'TRUE' or 'FALSE'): apply a nonlinear scale 
              to the colorbar?
          *'Curve' (str): name of scaling curve
          *'Scale Parameter' (float): scale parameter for asinh scaling
          *'Scale Parameter 2' and 'Scale Parameter 3' (floats): additional 
          scale parameters for asinhlin scaling
          IF THE FITS FILE IS 3D:
              'Cube Slice' (int): index of desired fits cube slice
          
    Outputs:
        prc_ims (dict): dictionary containing processed images. 
        The keys are the names of the disks, and the values are dictionaries 
        containing the corresponding processed image data (key: 'Image') and 
        scaling curve (key: 'Scaling').          
  """
  # Access global variables
  global old_prc_ims, old_prc_scales, old_imdat
  
  # Check if the dictionary has been updated since the last processing
  if imdat == old_imdat:
      print('All images already processed!')
      return old_prc_ims, old_prc_scales
  else:
      # Initialize return dicts
      prc_ims, prc_scales = {}, {}
      
      # For each image, normalize and return the image data
      for im in tqdm(imdat):
        # First, check if the image has already been processed.
        # Avoids redundant processing.
        if (im in old_prc_ims) and (im in old_imdat):
            if (imdat[im] == old_imdat[im]):
                # Don't go through processing; just use the old image
                prc_ims[im], prc_scales[im] = old_prc_ims[im], old_prc_scales[im]
                continue
        # Otherwise, process the image
        check_keys(imdat, im) # Check if the right keys are there before proceeding 
        prc_ims[im], prc_scales[im] = norm(imdat[im])
    
      #Save the current dictionaries
      old_prc_ims = prc_ims
      old_prc_scales = prc_scales
      old_imdat = imdat
      
      # Return the dictionary of normalized images. Use prc_scales to set
      # norm keyword when using plt.imshow().
      return prc_ims, prc_scales
