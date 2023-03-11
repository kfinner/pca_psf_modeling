#Quadrupole as translated from James Jee's quadrupole.pro
#by Kyle Finner
#May 2016

import numpy as np
import matplotlib.pyplot as plt

def quadrupole_uncertainty(img, xc, yc, sigma, rms=None, num_resamples=100):

    xsize = img.shape[1]
    ysize = img.shape[0]
    #print 'Your size:', xsize, ysize

    xa = np.array([np.linspace(0, xsize-1, xsize),]*xsize)
    ya = xa.T

    #Find the offset from the center of the map
    xa = xa - xc
    ya = ya - yc

    #Create a map of distance for each pixel from the center pixel
    rmap = np.sqrt(xa**2 + ya**2)

    #Create a Gaussian distribution map.
    gmap = np.exp(-(rmap**2)/(2.*sigma**2))
    gmap = gmap/gmap.sum()

    tot = np.sum(img*gmap)

    #Gaussian weighted quadrupole moment calculations
    q11 = np.sum(xa*xa*img*gmap)/tot
    q22 = np.sum(ya*ya*img*gmap)/tot
    q12 = np.sum(xa*ya*img*gmap)/tot

    q11_t, q12_t, q22_t = [],[],[]
    for i in range(num_resamples):
        np.random.seed()
        img2 = img + np.random.normal(0., rms, size=(xsize, ysize))

        #Add the values of the image after Gaussian weighting.
        tot = np.sum(img2*gmap)

        #Gaussian weighted quadrupole moment calculations
        q11_t.append(np.sum(xa*xa*img2*gmap)/tot)
        q22_t.append(np.sum(ya*ya*img2*gmap)/tot)
        q12_t.append(np.sum(xa*ya*img2*gmap)/tot)

    test = q11*q22 - q12**2

    try:
        e1 = (q11-q22) / (q11+q22 + 2.*np.sqrt(test))
        e2 = 2.*q12 / (q11+q22 + 2.*np.sqrt(test))
    except RuntimeWarning:
        print('The size of the object is', test)
        sys.exit()

    # This can be considered the size of the object.
    width = np.sqrt(q11 + q22)
    e = np.sqrt(e1**2 + e2**2)
    #print e1, e2, width

    return e1, e2, width, q11, q22, q12, np.std(q11_t), np.std(q22_t), np.std(q12_t)
