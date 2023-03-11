#SFIT
#translated to Python by Kyle Finner

import numpy as np

def sfit(z, degree):
    s = z.shape
    n2 = (degree+1) * (degree+2) // 2

    m = z.shape[0]
    x = z[:,0]
    y = z[:,1]
    zz = z[:,2]

    if n2 > m:
        print('Not enough data points for the degree you asked for.')

    ut = np.zeros((n2, m))
    k = 0
    # This gathers the terms needed for the polynomial fit.
    for i in range(degree+1):
        for j in range(degree+1):
            if i+j > degree:
                continue
            else:
                # ut contains arrays of the x,y positions that have been processed through the polynomial terms
                # ie. ut[0,:] contains an array of x**0 * y**0 --- ut[1,:] contains x**0 * y**1 and so on.
                ut[k, :] = (x**i * y**j).flatten()
                k += 1


    kk = np.dot(np.linalg.inv(np.dot(ut,ut.T)), ut) # ut

    # Multiply the kk terms by the flattened pca result
    # The flattened pca result contains the residual information built from all stars
    kx = np.dot(kk, zz.flatten())

    return kx
