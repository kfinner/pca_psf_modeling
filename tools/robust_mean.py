import numpy as np

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s<=m]

def robust_mean(data, m=2.):
    clipped = reject_outliers(data, m=m)
    return np.mean(clipped)
"""
def robust_mean(data, m=2.):
    n = len(data)
    med = np.median(data)
    abs_dev = np.abs(data - med)
    med_abs_dev = np.median(abs_dev) / 0.6745

    if med_abs_dev < 1.e-24:
        med_abs_dev = np.mean(abs_dev) / 0.8

    cutoff = m*med_abs_dev
    good_points = data[abs_dev <= cutoff]
    mean = np.mean(good_points)

    std = np.sqrt( np.sum( (good_points-mean)**2 ) / len(good_points) )
    print std
    #SC = m
    #if (SC <= 4.5) & (SC > 1.):
    #    std = std / (-0.15405+0.90723*SC-0.23584*SC**2+0.020142*SC**3)
    #print std
    cutoff = m*std
    good_points = data[abs_dev <= cutoff]
    mean = np.mean(good_points)

    return mean
"""
