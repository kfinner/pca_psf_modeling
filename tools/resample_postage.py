# Resample a postage stamp image to center it on a subpixel level
from scipy import interpolate
import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

def resample_postage(postage, xshift, yshift, s, padding):
    #print(xshift, yshift)
    # Define new array that we are going to sample the interpolated image onto.
    x_i = np.linspace(padding, s+padding-1, s) + xshift # Starts from 1 because we want to remove the +1 padding that we added before
    #print(x_i)
    y_i = np.linspace(padding, s+padding-1, s) + yshift
    #print(x_i,y_i)

    postage_size = np.linspace(0, postage.shape[0]-1, postage.shape[0]) # Postage size is 23x23 at this point. We will resample it at 21x21
    #print(postage_size, x_i, postage.shape)
    #sys.exit()
    #xx,yy = np.meshgrid(x_i,y_i)
    #n_postage = ndimage.map_coordinates(postage, [xx,yy], order=3)
    n_postage = interpolate.interp2d(postage_size, postage_size, postage, kind='cubic')
    n_postage = n_postage(x_i, y_i) # resample at shifted coordinates and 21x21
    #n_postage = griddata([x0.flatten(),y0.flatten()], postage.flatten(), (xx,yy))
    #n_postage = ndimage.shift(postage, [yshift,xshift])


    return n_postage#/np.sum(n_postage)
