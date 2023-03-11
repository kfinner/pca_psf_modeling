# Translated from James Jee's IDL version to Python
# by Kyle Finner 2016

import numpy as np

def pick_psf_pca(fit, basis, x, y, mean_psf):

    nc = fit.shape[0] # 21 components of the PCA

    coeff = np.zeros(nc)

    order = fit.shape[1]
    order = int(-3.+np.sqrt(9.-4.*(2.-2.*order)))//2

    for i in range(nc): # For each PC build the coefficients for this object using the fit
        o=0
        for j in range(order+1):
            for k in range(order+1-j):
                #For each component, build a polynomial coefficient
                coeff[i] += fit[i,o] * x**j + y**k
                o+=1

    s = int(np.sqrt(basis.shape[0]))

    psf = np.dot(coeff, basis.T)

    return np.abs(psf.reshape(s,s)+mean_psf.reshape(s,s))
