import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.wcs import WCS

def get_cartesian(xloc, yloc, major_axis, angle):
    #Define the semi-major and minor axis to 1 sigma
    xa = major_axis*np.cos(angle)
    ya = major_axis*np.sin(angle)

    # Determine the end points on the major axis.  The centroid is adjusted for the fit.
    x_ends = [xloc - xa, xloc + xa]
    y_ends = [yloc - ya, yloc + ya]

    #if xa < 200 and ya < 200:
    return x_ends, y_ends

def whisker_plot(bg_sources):#, massmap):

    #galaxies = ascii.read(bg_sources)
    galaxies = bg_sources

    #massmap = fits.open(massmap)[0]
    #massmap_wcs = WCS(massmap.header)
    #massmap = massmap.data
    x, y = np.linspace(0,1000,20), np.linspace(0,1000,20)
    xpoints, ypoints = [], []
    angles = []
    for i in range(len(x)):
        for j in range(len(y)):
            dist = np.sqrt((x[i] - galaxies['X'])**2 + (y[j] - galaxies['Y'])**2)
            try:
                within = galaxies[np.where(dist < 50)]
                me1 = np.mean(within['E1'])
                me2 = np.mean(within['E2'])
                e = np.sqrt(me1**2 + me2**2)

                angle = np.arctan2(me2, me1)/2.

                if (angle < 0) & (angle >= -np.pi):
                    angle += np.pi
                if angle < -np.pi:
                    print('Less than -pi!')
                    sys.exit()
                if angle > np.pi:
                    print('Greater than pi!')
                    sys.exit()
                angles.append(angle)


                e*=2000
                xpairs, ypairs = get_cartesian(x[i], y[j], e, angle)
                xpoints.append(xpairs)
                ypoints.append(ypairs)
            except IndexError:
                pass
    plt.hist(angles)
    """
    if (angle < 0) & (angle >= -np.pi):
        angle += np.pi
    if angle < -np.pi:
        print('Less than -pi!')
        sys.exit()
    if angle > np.pi:
        print('Greater than pi!')
        sys.exit()

    a = r*mag
    """


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    for x_ends, y_ends in zip(xpoints, ypoints):
        ax.plot(x_ends, y_ends, c='black', linewidth=1)
    plt.show()
    #plt.savefig('whisker_plot_abell2061.jpg')





#whisker_plot('/Volumes/hd16/MC2/abell2061_finished/analysis/bg_galaxy_r.dat')
