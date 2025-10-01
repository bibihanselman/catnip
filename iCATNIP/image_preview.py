'''
Helpers for image preview and processing.
'''

import numpy as np
import cv2
import math
import astropy.visualization as v
from astropy.visualization import ImageNormalize
from astropy.visualization.interval import ManualInterval
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
import astropy.stats as stats
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

######################

import plotly.express as px
import plotly.graph_objects as go

def edge_remove(im,imdat):
  """
  Sets pixel values along the edge of an image to nan.
  The length of the edges is determined by an adjustable parameter 'Crop',
  a key in imdat.

  Inputs:
      im (np array): fits file
      imdat (dict): dictionary containing processing parameters of the desired image
  Outputs:
      im (np array): input file with removed edges
  """
  # Defines the radius of pixels to remove from the image edge
  centery, centerx = imdat['Center']
  edge_crop = int((imdat['Crop'])/2)

  # Crop the image
  cropped_im = im[centery - edge_crop:centery + edge_crop + 1,centerx - edge_crop:centerx + edge_crop + 1]

  # Return the cropped image
  return cropped_im

######################

def mask_remove(im,imdat):
  """
  Sets pixel values within a given radius from the image center to nan.
  The radius is given by an adjustable parameter 'Radius', a key in imdat.

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

# Stretch dictionary (the stretch objects aren't JSON serializable)
def get_stretch(name, params):
    stretch_dict = {
        'linear': v.LinearStretch,
        'log': v.LogStretch,
        'sqrt': v.SqrtStretch,
        'squared': v.SquaredStretch,
        'asinh': v.AsinhStretch,
        'asinhlin': AsinhLinStretch
    }
    return stretch_dict[name](*params[name])

def update_image(im, imdat, title=None, plot=True):
    """
    Takes a .fits file path and returns a normalized and trimmed fits data file
    with an empty header.
    Inputs:
      imdat (dict): dictionary of image processing parameters for the desired image
    Output:
      norm_im (np array): normalized fits object with header that has the extreme values cut
    """

    # Remove edges and data behind the mask
    im = edge_remove(im,imdat)
    #im = mask_remove(im,imdat)
    #might add this back later... for FIGG galleries
    
    # Might add back later 11/21/24
    # if imdat['Mode Subtract']:
    #   mode = scipy.stats.mode(im,axis=None,nan_policy='omit')[0]
    #   im = im - mode
    #   im[im < 0] = np.nan
    
    # Find the median of the background by sigma clipping
    if imdat['Sigma'] > 0:
        bkg = stats.sigma_clip(im,imdat['Sigma'])
        im = im - np.ma.median(bkg)
        im[im < 0] = 0
        # sigma_clip = SigmaClip(sigma=imdat['σ'])
        # bkg_estimator = MedianBackground()
        # bkg = Background2D(im, imdat['Box Size'],
        #                    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
        #                    exclude_percentile=20)
        # im = im - bkg.background

    # Normalize the image
    bounds = imdat['Normalization']
    # Scale lower percentile of the image to 0
    im -= np.nanpercentile(im,float(bounds[0]))
    # Normalize the image to the upper percentile
    norm_im = im/np.nanpercentile(im,float(bounds[1]))

    # Check if we want to smooth
    if imdat['Smoothing'] > 0:
        # We smooth with a Gaussian kernel
        kernel = Gaussian2DKernel(x_stddev=float(imdat['Smoothing']))
        norm_im = convolve(norm_im, kernel, nan_treatment='interpolate')
        #Maybe add unsharp masking as an option? For finer features

    # Set colorbar bounds for scaling
    cb = imdat['Colorbar']
    cb = [float(cb[0]),float(cb[1])]

    # Colormap
    color = imdat['Color']
    
    # Stretching
    # Applying a scale to the data and artificially scaling colorbar values to match
    stretch = get_stretch(imdat['Stretch'], imdat['Parameters List'])
    if not plot:
        # Returning the normalized image and the scaling...
        normalize = ImageNormalize(norm_im, interval=ManualInterval(cb[0],cb[1]), stretch=stretch)
        return norm_im, normalize
    
    # Scale the image (thanks astropy docs)
    if stretch._supports_invalid_kw:
        norm_im = stretch(norm_im, out=norm_im, clip=False, invalid=0.0)
    else:
        norm_im = stretch(norm_im, out=norm_im, clip=False)
    # Set custom colorbar ticks
    cticks = np.linspace(cb[0],cb[1],6)
    cticktext = [f'{round(val,2)}' for val in cticks]
    ctickvals = stretch(cticks)
    # Scale bounds accordingly
    cb = stretch(cb)
    
    # Get arcsecond values for x and y ticks
    crop = int((imdat['Crop'])/2)
    tickvals = [i/2 * crop for i in range(5)]
    pixscale = imdat['Pixel Scale']
    distance = imdat['Distance']
    if imdat['Ticks'] == 'angles':
        ticktext = [f'{round((val - crop)*pixscale/1000,2)}"' for val in tickvals]
        xaxis_title = 'Relative RA'
        yaxis_title = 'Relative Dec'
    elif imdat['Ticks'] == 'pixels':
        ticktext = [str((int(val - crop))) for val in tickvals]
        xaxis_title = 'Pixel Offset (x)'
        yaxis_title = 'Pixel Offset (y)'
    else:
        ticktext = [f'{round((val - crop)*pixscale/1000 * distance,2)}' for val in tickvals]
        xaxis_title = 'Offset (au)'
        yaxis_title = 'Offset (au)'

    # Plot the image
    image = px.imshow(norm_im, color_continuous_scale = color,
                    labels = dict(color='Normalized Intensity'),
                    origin='lower', zmin = cb[0], zmax = cb[1])
    image.update_layout(
        title_text=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=dict(
            tickvals=tickvals,  # Specify the tick locations
            ticktext=ticktext  # Specify the tick labels
        ),
        yaxis=dict(
            tickvals=tickvals,  # Specify the tick locations
            ticktext=ticktext  # Specify the tick labels
        ),
        coloraxis_colorbar=dict(
            tickvals=ctickvals,
            ticktext=cticktext
        )
    )
                    
    # Return the plot of the image
    return image

######################

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
