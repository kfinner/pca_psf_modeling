#find_stars
#Translated from James Jee's IDL version
#by Kyle Finner April 2016
#This script finds stars that are within the threshold passed to it and returns their ID
import numpy as np
import tools.robust_sigma as rs
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table, vstack



def find_stars(cat, max_size=None, min_size=None, max_mag=None, min_mag=None, max_e=None, f=None, delmag=None, noflagcheck=False, w=0.2, log_name='star_log.txt'):
    n=30.
    #Calculate ellipticity based on the sextracted values.
    e = (cat['A_IMAGE'] - cat['B_IMAGE']) / (cat['A_IMAGE'] + cat['B_IMAGE'])
    #Make an array of 0s of size 10
    n_star = np.zeros(int(n))

    #Creates an 11 valued array from min to max of equal steps.
    flux_radius = (np.arange(n+1.)/n)*(max_size-min_size)+min_size

    #t is the sorted argument by flux_max high to low.
    t = cat['FLUX_MAX'].argsort()[::-1]

    #Set the maximum flux to the 20th highest flux star.
    flux_max = cat[t[1]]['FLUX_MAX']

    #Iterate through each radius level and determine how many objects are within that level.
    for i in range(int(n)):
        a = np.where( (cat['FLUX_RADIUS'] > (flux_radius[i]-w)) & (cat['FLUX_RADIUS'] < (flux_radius[i]+w)) & (cat['MAG_ISO'] > min_mag) & (cat['MAG_ISO'] < max_mag) & (e < max_e) & (cat['FLUX_MAX'] < f*flux_max))

        #Save the number of stars that were found.
        if a[0].size != 0:
            n_star[i] = a[0].size

    #Determine the flux_radius that found the most number of stars and second most.
    m1 = np.max(n_star)
    i = np.argmax(n_star)
    n_star[i] = 0. # set the one that had the most to 0.
    #Determine the constraints that found the 2nd most objects
    m2 = np.max(n_star)
    j = np.argmax(n_star)
    #If the flux_radius that found the most number of objects is greater than the flux_radius that had the 2nd most and if 2nd place had more than half as many stars then use the second most numerous flux_radius from now on. (ie. the larger PSF)

    try:
        mratio = float(m2)/float(m1)
    except ZeroDivisionError:
        mratio = float(m2)/1.

    if (i > j) & (mratio > 0.75) & (m2 > 20.):
        i=j
    #Select sample (a) based on either the most detections or 2nd most depending on previous condition.
    #So this does nothing if i=i.  But it grabs j if i=j now.
    a = np.where( (cat['FLUX_RADIUS'] > (flux_radius[i]-w)) & (cat['FLUX_RADIUS'] < (flux_radius[i]+w)) & (cat['FLAGS'] == 0) & (cat['MAG_ISO'] > min_mag) & (e < max_e) & (cat['MAG_ISO'] < max_mag) )# & (cat['FLUX_MAX'] < f*flux_max)) #

    star_pool = Table([cat[a]['MAG_ISO'], cat[a]['FLUX_RADIUS'], e[a]], names=['MAG_ISO', 'FLUX_RADIUS', 'ELLIPTICITY'])
    star_pool.add_row([len(a[0]), '0000000', '00000000'])
    ascii.write(star_pool, log_name , overwrite=True)

    min_mag = np.min(cat[a]['MAG_ISO'])*f
    print('Min mag', min_mag)
    #Set the max_mag (faintest object) to the magnitude of object in the sample that we have now selected + delmag.
    max_mag = np.min(cat[a]['MAG_ISO'])+delmag # This is the lower y value in the plot

    #Find the median flux_radius of our sample.
    flux_radius_med = np.median(sigma_clip(cat[a]['FLUX_RADIUS'], 3.))
    #Find the standard dev of the sample.
    flux_radius_sig = rs.robust_sigma(cat[a]['FLUX_RADIUS'])

    lin_m, lin_b = np.polyfit(cat[a]['MAG_ISO'], cat[a]['FLUX_RADIUS'],1)

    if lin_m > 0.:
        lin_b = np.median(cat[a]['FLUX_RADIUS'])
        lin_m = 0.

    c = np.where( (cat['FLUX_RADIUS'] < lin_m*cat['MAG_ISO'] + lin_b + w) & (cat['FLUX_RADIUS'] > lin_m*cat['MAG_ISO'] + lin_b - w) & (cat['MAG_ISO'] < max_mag) & (cat['MAG_ISO'] > min_mag) & (e < max_e) )#& (cat['FLUX_MAX'] < f*flux_max))
    star_pool = vstack([star_pool, Table([cat[c]['MAG_ISO'], cat[c]['FLUX_RADIUS'], e[c]], names=['MAG_ISO', 'FLUX_RADIUS', 'ELLIPTICITY'])])
    star_pool.add_row([len(c[0]), '0000000', '00000000'])
    ascii.write(star_pool, log_name , overwrite=True)

    return c, lin_m, lin_b, max_e
