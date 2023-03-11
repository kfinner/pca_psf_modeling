#find_stars
#Translated from James Jee's IDL version
#by Kyle Finner April 2016

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.utils import lazyproperty
from termcolor import colored
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings("ignore", category=UserWarning, module='numpy')
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter('ignore', category=AstropyWarning)


class pcomp(object):
    """Replicates the IDL ``PCOMP()`` function.

    The attributes of this class are all read-only properties, implemented
    with :class:`~astropy.utils.decorators.lazyproperty`.

    Parameters
    ----------
    x : array-like
        A 2-D array with :math:`N` rows and :math:`M` columns.
    standardize : :class:`bool`, optional
        If set to ``True``, the input data will have its mean subtracted off
        and will be scaled to unit variance.
    covariance : :class:`bool`, optional.
        If set to ``True``, the covariance matrix of the data will be used for
        the computation.  Otherwise the correlation matrix will be used.

    Notes
    -----

    References
    ----------
    http://www.exelisvis.com/docs/PCOMP.html

    Examples
    --------
    """

    def __init__(self, x, standardize=False, covariance=False):
        from scipy.linalg import eigh
        if x.ndim != 2:
            raise ValueError('Input array must be two-dimensional')
        no, nv = x.shape
        self._nv = nv
        if standardize:
            xstd = x - np.tile(x.mean(0), no).reshape(x.shape)
            s = np.tile(xstd.std(0), no).reshape(x.shape)
            self._array = xstd/s
            self._xstd = xstd
        else:
            self._array = x
            self._xstd = None
        self._standardize = standardize
        if covariance:
            self._c = np.cov(self._array, rowvar=0)
        else:
            self._c = np.corrcoef(self._array, rowvar=0)
        self._covariance = covariance
        #
        # eigh is used for symmetric matrices
        #
        evals, evecs = eigh(self._c)
        #
        # Sort eigenvalues in descending order
        #
        ie = evals.argsort()[::-1]
        self._evals = evals[ie]
        self._evecs = evecs[:, ie]
        #
        # If necessary, add code to fix the signs of the eigenvectors.
        # http://www3.interscience.wiley.com/journal/117912150/abstract
        #
        return

    @lazyproperty
    def coefficients(self):
        """(:class:`~numpy.ndarray`) The principal components.
        These are the coefficients of `derived`.
        Basically, they are a re-scaling of the eigenvectors.
        """
        return self._evecs * np.tile(np.sqrt(self._evals), self._nv).reshape(
            self._nv, self._nv)

    @lazyproperty
    def derived(self):
        """(:class:`~numpy.ndarray`) The derived variables.
        """
        derived_data = np.dot(self._array, self.coefficients)
        if self._standardize:
            derived_data += self._xstd
        return derived_data

    @lazyproperty
    def variance(self):
        """(:class:`~numpy.ndarray`) The variances of each derived variable.
        """
        return self._evals/self._c.trace()

    @lazyproperty
    def eigenvalues(self):
        """(:class:`~numpy.ndarray`) The eigenvalues.
        There is one eigenvalue for each principal component.
        """
        return self._evals

def pca_stars(outname, residual, mean_psf, coordinates, n_pcs=21, imgheader='none', shape_measurements=None):
    print('-------------', n_pcs, '-------------')
    outfile = outname+'.pca'

    ############# PCA #######################
    print(colored('Doing PCA on %s' % outname, 'green'))
    res = pcomp(residual, covariance=True)#, standardize=True) N stars by M pixels

    result = res.derived[:,0:n_pcs]
    eigenvalues = res.eigenvalues[0:n_pcs]
    eigenvalues[eigenvalues < 1e-8] = 1e-8
    eigenvectors = res.coefficients[:,0:n_pcs] / eigenvalues

    egnvs = res.eigenvalues[res.eigenvalues > 1e-12]

    plt.close('all')
    plt.plot(np.arange(len(egnvs)), np.cumsum(egnvs)/np.sum(egnvs))
    #plt.yscale('log')
    plt.ylabel('Variance')
    plt.xlabel('Component')
    plt.xlim(0,len(egnvs))
    #plt.axvline(x=residual.shape[0], label='N stars', color='black', linestyle='--')
    plt.axvline(x=n_pcs, label=f'{n_pcs} components', color='black', linestyle='-.')
    plt.legend(loc=1, prop={'size': 12})
    plt.savefig(outname+'_cumvar_PCs.pdf', bbox_inches='tight')
    plt.close('all')

    plt.close('all')
    plt.plot(np.arange(len(egnvs)), egnvs)
    plt.yscale('log')
    plt.ylabel('Variance')
    plt.xlabel('Component')
    plt.xlim(0,len(egnvs))
    #plt.axvline(x=residual.shape[0], label='N stars', color='black', linestyle='--')
    plt.axvline(x=n_pcs, label=f'{n_pcs} components', color='black', linestyle='-.')
    plt.legend(loc=1, prop={'size': 12})
    plt.savefig(outname+'_var_PCs.pdf', bbox_inches='tight')
    plt.close('all')

    print('Mean PSF shape', mean_psf.shape)
    print('Eigenvectors shape', eigenvectors.shape)
    print('Result shape', result.shape)
    print('Coordinates shape', coordinates.shape)
    if imgheader == 'none':
        hdu = fits.PrimaryHDU()
        hdu.header['Nstars'] = len(coordinates[0])
        imgheader = hdu.header

    if shape_measurements:
        psflist = fits.HDUList([fits.PrimaryHDU(mean_psf, header=imgheader),fits.ImageHDU(eigenvectors),fits.ImageHDU(result),fits.ImageHDU(coordinates), fits.BinTableHDU(shape_measurements)])
    else:
        psflist = fits.HDUList([fits.PrimaryHDU(mean_psf, header=imgheader),fits.ImageHDU(eigenvectors),fits.ImageHDU(result),fits.ImageHDU(coordinates)])
    psflist.writeto(outfile, overwrite=True)
