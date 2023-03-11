# by Kyle Finner April 2016

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.table import Table, hstack
from matplotlib.colors import LogNorm
from termcolor import colored
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings("ignore", category=UserWarning, module='numpy')
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter('ignore', category=AstropyWarning)


from tools.quadrupole import first_moment
from tools.plot_needles import plot_needles
from tools.resample_postage import resample_postage
from matplotlib.patches import Ellipse

def create_size_mag(root, cat, star_sample, min_mag, max_mag, mag_zp):
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)

    ax.scatter(cat['FLUX_RADIUS'], cat['MAG_APER'][:,4]+mag_zp, alpha=0.5, marker='.', color='black')
    ax.scatter(star_sample['FLUX_RADIUS'], star_sample['MAG_APER'][:,4]+mag_zp, c='red', marker='*', s=150)
    ax.set_ylim(max_mag+3+mag_zp,min_mag-3+mag_zp)
    ax.set_xlim(0,10)
    ax.set_xlabel(r'$\tt{FLUX\_RADIUS}$ [pixels]', fontsize=30)
    ax.set_ylabel(r'$\tt{MAG\_ISO}$', fontsize=30)
    #plt.title(root, fontsize=30)
    ax.text(0.4, 0.95, 'Number stars chosen: %s' % len(star_sample), fontsize=30, transform=ax.transAxes)
    ax.text(0.4, 0.90, 'Median half-light rad: %s' % round(np.median(star_sample['FLUX_RADIUS']),3), fontsize=30, transform=ax.transAxes)
    ax.text(0.4, 0.85, 'Median FWHM: %s' % round(np.median(star_sample['FWHM_IMAGE']),3), fontsize=30, transform=ax.transAxes)
    ax.tick_params(axis='x',which='major', size=10)
    ax.tick_params(axis='y',which='major', size=10)
    plt.savefig(root+'_all_star.jpg', bbox_inches='tight')

def test_segmentation(seg, num):
    #plt.imshow(seg, origin='lower')

    if seg.shape[0] >= 15:
        inner = seg[seg.shape[1]//2-7:seg.shape[1]//2+8, seg.shape[0]//2-7:seg.shape[0]//2+8]
    else:
        return 0.

    inner[inner == 0] = num
    #(x for x in xyz if x not in a)
    n = np.array([1 for p in inner.flatten() if p != num])
    return n.sum() / len(inner.flatten()) # percent of pixels in the inner region that do not belong to the object.

def test_flux(post):
    xx,yy = np.meshgrid(np.arange(post.shape[0]),np.arange(post.shape[1]))
    xx -= post.shape[0]//2
    yy -= post.shape[1]//2
    rr = np.sqrt(xx**2+yy**2)
    core = np.copy(post)
    core[rr > 3] = 0
    outer = np.copy(post)
    outer[rr < 3] = 0

    return core.sum()/outer.sum()


def get_postage_images(n, x, y, s, w, star_sample, img, max_e, min_e, root, seg, rms, cat, mag_zp, cenmax, sigma, pixscale):

    xarr = np.array([np.linspace(0, s-1, s),]*s) # Create a 31x31 array of pixels
    yarr = xarr.T

    nt_xc, nt_yc = [],[] # Centroid tracker
    xs, ys, e1s, e2s, es, phis = [],[],[],[],[],[] # Shape tracker
    widths = [] # Size measurement tracker
    seg_tests, star_ids = [],[]

    data = np.zeros([n, s**2]) # N stars by M pixels

    padding = 1

    bad_objs = 0
    good_stars_mask = np.zeros(len(star_sample), dtype=bool)
    good_star_shapes = Table(names=('E1', 'E2', 'WIDTH'))
    good_star_postages,shifted_star_postages, all_star_postages = [],[],[]
    print('Testing %s objects.' % len(star_sample))
    for i, star in enumerate(star_sample): # for each star

        x, y, = star['XWIN_IMAGE']-1., star['YWIN_IMAGE']-1.
        #Find center coordinate of the star
        cx = int(round(x,0)) # Use int to remove decimal place.
        cy = int(round(y,0)) # The fraction will be shifted back during resample.

        #Calculate the edges of the postage image
        x1 = cx - w - padding # Lets take a bit larger so that we have room to interpolate
        x2 = cx + w + padding + 1 # +1 for room and +1 more because python slices from x1 to x2-1 .
        y1 = cy - w - padding
        y2 = cy + w + padding + 1

        postage = img[y1:y2, x1:x2]-star['BACKGROUND'] # This should be 23x23 because of +1 padding.

        rms_post = np.median(rms[y1:y2, x1:x2])
        # Find the shift from the int center to actual center
        xshift = x - cx
        yshift = y - cy
        if postage.shape[0] != postage.shape[1]:
            continue
        n_postage = resample_postage(postage, xshift, yshift, s, padding)#.flatten()
        n_postage /= n_postage.sum() # Normalize it.

        n_xc, n_yc = first_moment(n_postage, sigma=1) # The centroid
        #print(sigma)
        e1, e2, width, q11, q22, q12, q11err, q22err, q12err = quadrupole_uncertainty(n_postage, s//2, s//2, sigma, rms=rms_post) # use the quadrupole moment to find ellipticity, s/2 should be integer

        e = np.sqrt(e1**2 + e2**2)
        phi = np.arctan2(e2, e1)/2.

        seg_test = test_segmentation(seg[y1:y2, x1:x2], star['NUMBER'])
        flux_test = test_flux(n_postage)

        star_ids.append(star['NUMBER'])
        nt_xc.append(n_xc)
        nt_yc.append(n_yc)
        es.append(e)
        phis.append(phi)
        seg_tests.append(seg_test)

        all_star_postages.append(n_postage)


        if (e < max_e) & (np.abs(n_xc) < cenmax) & (np.abs(n_yc) < cenmax):

            good_star_postages.append(postage)
            shifted_star_postages.append(n_postage)
            data[i,:] = n_postage.flatten()

            good_star_shapes.add_row([e1,e2,width, q11, q22, q12, q11err, q22err, q12err])
            good_stars_mask[i] = 1

    good_star_sample = star_sample[good_stars_mask]
    data = data[good_stars_mask,:]
    good_stars_cat = hstack([good_star_sample, good_star_shapes])
    good_stars_cat.write(root.replace('.fits', '_star_properties.cat'), format='fits', overwrite=True)
    plot_needles(good_stars_cat['XWIN_IMAGE'], good_stars_cat['YWIN_IMAGE'], good_stars_cat['E1'], good_stars_cat['E2'], 2000, outimage=root+'.ell.jpg')

    ''' PLOTTING '''
    print(colored('%s stars final selection.' % len(good_star_shapes), 'yellow', attrs=['bold']))
    print('Mean centroid: ', round(np.mean(nt_xc),3), round(np.mean(nt_yc),3))
    print('Scatter in centroid: ', round(np.std(nt_xc),3), round(np.std(nt_yc),3))

    plt.figure(figsize=(10,10))
    plt.imshow(img, origin='lower', vmin=0, vmax=1, cmap='gray_r')
    plt.scatter(good_stars_cat['XWIN_IMAGE'], good_stars_cat['YWIN_IMAGE'], facecolor='none', edgecolor='red', linewidth=2, s=50)
    for st in good_stars_cat:
        plt.text(st['XWIN_IMAGE']+5, st['YWIN_IMAGE']+5, st['NUMBER'])
    plt.title(root, fontsize=10)
    plt.xlabel('XWIN_IMAGE', fontsize=10)
    plt.ylabel('YWIN_IMAGE', fontsize=10)
    plt.savefig(root.replace('.fits', '_stars.jpg'), bbox_inches='tight', dpi=200)

    plt.clf()
    widths = np.array(widths)
    plt.imshow(img, origin='lower', vmin=0, vmax=100, cmap='gray_r')
    cbar = plt.scatter(good_stars_cat['XWIN_IMAGE'], good_stars_cat['YWIN_IMAGE'], c=good_stars_cat['WIDTH']*pixscale, cmap='jet', edgecolor='none', s=100)#, vmin=np.median(widths*pixscale)-2.*np.std(widths*pixscale), vmax=np.median(widths*pixscale)+2.*np.std(widths*pixscale))
    plt.colorbar(cbar)
    plt.savefig(root.replace('.fits', '_star_size.jpg'), bbox_inches='tight')


    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    ax.scatter(cat['FLUX_RADIUS'], cat['MAG_ISO']+mag_zp, alpha=0.5, marker='.', color='black')
    ax.scatter(good_stars_cat['FLUX_RADIUS'], good_stars_cat['MAG_ISO']+mag_zp, c='red', marker='*', s=150)
    ax.invert_yaxis()#ax.set_ylim(5.,-12.)
    #ax.set_ylim(ymin=2)
    ax.set_xlim(0,10)
    ax.set_ylim(28,12)
    ax.set_xlabel(r'$\tt{FLUX\_RADIUS}$ [pixels]', fontsize=30)
    ax.set_ylabel(r'$\tt{MAG\_ISO}$', fontsize=30)
    ax.text(0.4, 0.95, 'Number stars chosen: %s' % len(good_stars_cat['FLUX_RADIUS']), fontsize=30, transform=ax.transAxes)
    ax.text(0.4, 0.90, 'Median half-light rad: %s pix' % round(np.median(good_stars_cat['FLUX_RADIUS']),3), fontsize=30, transform=ax.transAxes)
    ax.text(0.4, 0.85, 'Median FWHM: %s pix' % round(np.median(good_stars_cat['FWHM_IMAGE']),3), fontsize=30, transform=ax.transAxes)
    ax.tick_params(axis='x',which='major', size=10)
    ax.tick_params(axis='y',which='major', size=10)
    plt.savefig(root+'_good_star.jpg', bbox_inches='tight')

    return data, good_star_postages, good_stars_cat, shifted_star_postages, es, nt_xc, nt_yc, seg_tests, phis, all_star_postages, star_sample

def get_postages(root, min_size=1.5, max_size=4.3, max_e=0.15, min_e=0.0, min_mag=-10.5, max_mag=-6., flux_limit=0.9, del_mag=3.0, width=0.2, noflagcheck=False, nopca=False, tight_select=False, find_grid=40., cenmax=0.1, sigma=3., mag_zp=0.):

    catname = root.replace('.fits', '.cat')
    starname = root.replace('.fits', '.star')
    cat = fits.open(catname)[2].data

    try:
        img = fits.open(root)[1].data
        imgheader = fits.open(root)[1].header
        seg = fits.open(root.replace('.fits', '_seg.fits'))[1].data
        rms = fits.open(root.replace('sci.fits', 'rms.fits'))[1].data
    except IndexError:
        img = fits.open(root)[0].data
        imgheader = fits.open(root)[0].header
        seg = fits.open(root.replace('.fits', '_seg.fits'))[0].data
        rms = fits.open(root.replace('sci.fits', 'rms.fits'))[0].data

    pixscale = np.sqrt(imgheader['CD1_1']**2 + imgheader['CD2_1']**2)*3600
    #sigma = sigma / pixscale # Sigma is deinfed in arcsec. Change it to pixels.

    print(colored('===== Get Postages ===== ', 'blue'))

    star_sample = Table(fits.open(starname)[1].data) # Open up the catalog of stars.

    star_id = star_sample['NUMBER']
    sx = star_sample['XWIN_IMAGE']
    sy = star_sample['YWIN_IMAGE']
    x,y = sx - 1., sy - 1. # Subtract 1 from SE coordinates to put in python

    s = 31 # Size of the postage image
    w = s//2 # Offset between center pixel and postage edge

    data, good_star_postages, good_stars_cat, shifted_star_postages, es, nt_xc, nt_yc, seg_tests, phis, all_postages, all_star = get_postage_images(len(star_id), x, y, s, w, star_sample, img, max_e, min_e, root, seg, rms, cat, mag_zp, cenmax, sigma, pixscale)

    ''' Accumulate the data '''
    coordinates = np.asarray([good_stars_cat['XWIN_IMAGE']-1., good_stars_cat['YWIN_IMAGE']-1.]) # actual coordinates for each star

    data = np.asarray(data)
    mean_psf = np.median(data, axis=0) #Find median for each pixel (sum of each pixel[i])/113
    mean_psf_arr = np.tile(mean_psf, (data.shape[0],1))# Create a N stars by the 441 mean values array.
    data_diff = data - mean_psf_arr # Subtract the mean values from the data
    if len(coordinates) > 3:
        # Set pixels that are outliers to the mean value.
        for pixel in range(s*s):
            d = data_diff[:,pixel]
            std = rs.robust_sigma(d)
            outlier = np.where(np.abs(d) > 3.*std)
            data[outlier,pixel] = mean_psf[pixel]

    # Recalculate the mean and recreate the mean array.
    mean_psf = np.median(data, axis=0)
    mean_psf_arr = np.tile(mean_psf, (data.shape[0],1))
    data_diff = data - mean_psf_arr # 441 rows, 61 columns (ie. table with 61 vectors/columns)

    hdu = fits.PrimaryHDU()
    hdu.header['NSTARS'] = data_diff.shape[0]
    hdu.header['Pixels per star'] = data_diff.shape[1]

    hdu1 = fits.ImageHDU(data_diff)
    hdu2 = fits.ImageHDU(mean_psf)
    hdu3 = fits.ImageHDU(coordinates)
    hdu_stack = fits.HDUList([hdu, hdu1, hdu2, hdu3])
    hdu_stack.writeto(root.replace('.fits', '_residual.fits'), overwrite=True)

    phdu = fits.PrimaryHDU()
    phdu.header['NSTARS'] = len(good_stars_cat)
    phdu.header['HDU1'] = 'Raw Postages'
    phdu.header['HDU2'] = 'Interpolated/Shifted Postages'
    phdu.header['HDU3'] = 'Residual postages'
    phdu.header['HDU4'] = 'Sigma Clipped Postages'
    phdu.header['HDU5'] = 'Mean PSF'
    phdu.header['HDU6'] = 'SExtractor/Shape Properties'

    hdu = fits.ImageHDU(good_star_postages)
    hdu1 = fits.ImageHDU(shifted_star_postages)
    hdu2 = fits.ImageHDU(data_diff.reshape(len(good_stars_cat),s,s))
    hdu3 = fits.ImageHDU(data.reshape(len(good_stars_cat),s,s))
    hdu4 = fits.ImageHDU(mean_psf.reshape(s,s))
    hdu5 = fits.BinTableHDU(good_stars_cat)

    hduwrite = fits.HDUList([phdu, hdu, hdu1, hdu2, hdu3, hdu4, hdu5])
    hduwrite.writeto(root.replace('.fits', '_star_postages.fits'), overwrite=True)

    # Write a postage stamp JPG for perusing.
    for i, (gs, gs_id) in enumerate(zip(all_postages, all_star)):
        if gs_id['NUMBER'] in good_stars_cat['NUMBER']:
            its_good = 'good'
        else:
            its_good = 'bad'
        plt.close('all')
        import matplotlib.colors as colors
        fig, ax = plt.subplots(1,1)
        scaled_fig = (gs-gs.min())/(gs.max()-gs.min())
        if np.median(scaled_fig) < scaled_fig[w,w]:
            ax.imshow(scaled_fig, origin='lower', cmap='jet',norm=colors.LogNorm(vmin=np.median(scaled_fig), vmax=scaled_fig[w,w]/1.1))
        else:
            ax.imshow(scaled_fig, origin='lower', cmap='bone')

        ax.scatter(15+nt_xc[i],15+nt_yc[i], marker='x', color='black')
        ax.axvline(15, color='black', linewidth=0.5)
        ax.axhline(15, color='black', linewidth=0.5)
        ellipse = Ellipse((15+nt_xc[i],15+nt_yc[i]), width=5, height=5.*(1-es[i])/(1+es[i]), angle=phis[i]*180/np.pi)
        ax.add_artist(ellipse)
        ellipse.set_facecolor('none')
        ellipse.set_edgecolor('white')
        if np.isnan(es[i]):
            continue
        else:
            plt.title('%s star!, E: %s' % (its_good, round(es[i],2)))
        fname = 'star_postages'+'/%s_%s_star.jpg' % (root.replace('_single_sci.fits', ''), gs_id['NUMBER'])
        if not os.path.exists('star_postages'):
            os.mkdir('star_postages')
        try:
            plt.savefig(fname, bbox_inches='tight')
        except ValueError:
            print(colored('PlotWARNING: Star postage %s cannot be saved.' % gs_id['NUMBER'], 'red'))
        #plt.show()
        plt.close('all')
    return len(good_stars_cat), good_stars_cat['IDs'], np.median(np.sqrt(good_stars_cat['E1']**2+good_stars_cat['E2']**2)), np.median(good_stars_cat['FLUX_RADIUS'])
