# fit_psf_pca
# Converted from James Jee's IDL version to Python
# by Kyle Finner
import sys
import numpy as np
import itertools
from psf_modeling.sfit import sfit
from astropy.io import fits

def fit_psf_pca(infile, order, head):
    #Get data from the PCA file
    pcafile = fits.open(infile)

    mean_psf = pcafile[0].data # 441 x N stars : N copies of the mean_psf
    basis = pcafile[1].data # 441 x 21 ie. 21 eigenvectors
    pca_result = pcafile[2].data # N stars x 21 : the residual projected onto the PCs
    coordinates = pcafile[3].data # 2 x N stars
    # Need at least 5 samples and if there aren't 15 then do a 2nd order fit.
    if coordinates.shape[1] < 15:
        print('Very few stars in this image!')

    nc = basis.shape[1] # 21 principal components
    fit = np.zeros((nc, (order+1)*(order+2)//2)) #21 by 16

    for i in range(nc): # For each principal component we fit a polynomial across all stars in the frame.
        xs, ys = coordinates[0,:], coordinates[1,:]
        d = pca_result[:,i] # d is all star data for the i'th princ. comp.
        dat = np.stack((xs, ys, d), axis=1)
        fit[i,:] = sfit(dat, 3) # fit for each component

    return fit, basis, mean_psf
