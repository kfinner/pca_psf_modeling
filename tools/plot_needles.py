#plot_needle2.py as translated from James Jee's plot_needle2.pro
#by Kyle Finner
#May 2016

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
from matplotlib.colors import LogNorm

#Given a set of coordinates, a length, and an angle, return the coordinates of endpoints for line
def get_cartesian(xloc, yloc, major_axis, angle):
    #Define the semi-major and minor axis to 1 sigma
    xa = major_axis*np.cos(angle)
    ya = major_axis*np.sin(angle)

    return xloc-xa, xloc+xa, yloc-ya, yloc+ya

def plot_needles(xc, yc, ex, ey, mag, colorimage=None, colorimage_wcs=None, phi_scatter=None, outimage='.ell.png', flags=None, plotellipses=False, xlim='none', ylim='none', mags='none', all_x='none', all_y='none'):
    xpoints, ypoints = [],[]
    ells, ells2 = [], []

    r = np.sqrt(ex**2 + ey**2)
    angle = np.arctan2(ey, ex)/2.
    a = r*mag

    x0,x1,y0,y1 = get_cartesian(xc, yc, a, angle)

    fig1 = plt.figure()#figsize=(12,12))
    ax1 = fig1.add_subplot(1,1,1)

    if not all_x == 'none':
        for ras, decs in zip(all_x, all_y):
            for i in range(3):
                plt.plot([ras[i],ras[i+1]], [decs[i],decs[i+1]], color='gray', alpha=0.1)
            plt.plot([ras[3],ras[0]], [decs[3],decs[0]], color='gray', alpha=0.1)

    for i, (x0,x1,y0,y1) in enumerate(zip(x0,x1,y0,y1)):

        ax1.plot([x0,x1], [y0,y1], c='black', linewidth=1)
        if not mags == 'none':
            ax1.text(x1, y1, '%s' % round(mags[i],1) , fontsize=16)




    mean_x0, mean_x1, mean_y0, mean_y1 = get_cartesian(ax1.get_xlim()[1]/2, ax1.get_ylim()[1]/2, 0.05*mag, 0)
    ax1.plot([mean_x0,mean_x1], [mean_y0, mean_y1], c='red', linewidth=1, label='5% shear')
    if not xlim=='none':
        plt.xlim(0,xlim)
    if not ylim=='none':
        plt.ylim(0,ylim)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(outimage, dpi=300)
    plt.close('all')
