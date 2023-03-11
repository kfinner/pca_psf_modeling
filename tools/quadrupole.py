#Quadrupole as translated from James Jee's quadrupole.pro
#by Kyle Finner
#May 2016

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def first_moment(img, sigma=0):
    xsize = img.shape[1]
    ysize = img.shape[0]
    xa = np.array([np.linspace(0, xsize-1, xsize),]*xsize)
    ya = np.array([np.linspace(0, ysize-1, ysize),]*ysize).T

    xa = xa - xsize // 2
    ya = ya - ysize // 2

    if sigma == 0:
        gmap = 1
    else:
        #Create a map of distance for each pixel from the center pixel
        rmap = np.sqrt(xa**2 + ya**2)
        #Create a Gaussian distribution map.
        gmap = np.exp(-(rmap**2)/(2.*sigma**2))
        gmap = gmap/gmap.sum()

    return (xa*img*gmap).sum() / (img*gmap).sum(), (ya*img*gmap).sum() / (img*gmap).sum()

def quadrupole(img, xc, yc, sigma):
    #print(sigma)
    xsize = img.shape[1]
    ysize = img.shape[0]
    #print 'Your size:', xsize, ysize

    xa = np.array([np.linspace(0, xsize-1, xsize),]*xsize)
    ya = np.array([np.linspace(0, ysize-1, ysize),]*ysize).T


    #Find the offset from the center of the map
    xa = xa - xc
    ya = ya - yc

    if sigma == 0:
        gmap = 1

    else:
        #Create a map of distance for each pixel from the center pixel
        rmap = np.sqrt(xa**2 + ya**2)
        #Create a Gaussian distribution map.
        gmap = np.exp(-(rmap**2)/(2.*sigma**2))
        gmap = gmap/gmap.sum()




    #Add the values of the image after Gaussian weighting.
    tot = np.sum(img*gmap)

    q11 = np.sum(xa*xa*img*gmap)/tot
    q22 = np.sum(ya*ya*img*gmap)/tot
    q12 = np.sum(xa*ya*img*gmap)/tot
    q21 = np.sum(ya*xa*img*gmap)/tot

    test = q11*q22 - q12**2

    #Find ellipticities from the quad moments.
    e1 = (q11-q22) / (q11+q22 + 2.*np.sqrt(test))
    e2 = 2.*q12 / (q11+q22 + 2.*np.sqrt(test))

    # This can be considered the size of the object.
    width = np.sqrt(q11 + q22)
    #e = np.sqrt(e1**2 + e2**2)

    return e1, e2, width, q11, q22, q12
