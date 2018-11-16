"""Main ALMA Ceres data processing module
"""

import json
from os import path
import numpy as np
from astropy import units, table, constants, time
from astropy.io import fits
import spiceypy as spice
from . import utils


# Ceres constants
class Ceres(object):
    # body shape and pole based on DLR RC3 shape model
    ra = 482.64 * units.km
    rb = 480.60 * units.km
    rc = 445.57 * units.km
    r = np.sqrt((ra+rb)*rc/2)
    pole = (291.41, 66.79)
    GM = 62.68 * units.Unit('km3/s2')
    M = (GM/constants.G).decompose()


class ALMACeresImageMetaData(utils.MetaData):
    """Meta data class for Ceres ALMA images
    """
    def collect(self, datafile, spicekernel=None, **kwargs):
        super().collect(datafile)
        hdr = fits.open(datafile)[0].header
        self.bmaj = hdr['bmaj']*3600_000.
        self.bmin = hdr['bmin']*3600_000.
        self.bpa = hdr['bpa']*3600_000.
        self.xscl = hdr['cdelt1']*3600_000.
        self.yscl = hdr['cdelt2']*3600_000.
        for k, v in kwargs.items():
            setattr(self, k, v)
        if hasattr(self, 'utc_start') and hasattr(self, 'utc_stop'):
            utc1 = getattr(self, 'utc_start')
            utc2 = getattr(self, 'utc_stop')
            self.utc_mid = (time.Time(utc1)+(time.Time(utc2)-time.Time(utc1))/2.).isot
            sub = self.calc_geometry(spicekernel=spicekernel)
            sub.remove_column('Time')
            for k in sub.colnames:
                setattr(self, k, sub[k][0])

    def calc_geometry(self, spicekernel=None):
        """Calculate observing geometry
        """
        if spicekernel is not None:
            if not is_iterable(spicekernel):
                spicekernel = [spicekernel]
            dummy = [spice.furnsh(k) for k in spicekernel]
        sub = utils.subcoord(self.utc_mid,'ceres')
        if spicekernel is not None:
            dummy = [spice.unload(k) for k in spicekernel]
        return sub


def aspect(filenames, utc1, utc2, outfile=None, spicekernel=None, **kwargs):
    '''Collect aspect data for ALMA Ceres images

    filenames : str or array_like(str)
      Name (with full path) of ALMA Ceres images
    utc1, utc2 : str array_like(str)
      Start and end time of each file in `filenames`
    outfile : str, optional
      Name of output file
    spicekernel : str or array_like(str)
      The filenames of SPICE kernels to be loaded
    '''
    mc = utils.MetaDataList(ALMACeresImageMetaData)
    if spicekernel is not None:
        spice.furnsh(spicekernel)
    mc.collect(filenames, utc_start=utc1, utc_stop=utc2, **kwargs)
    if spicekernel is not None:
        spice.unload(spicekernel)
    tbl = mc.to_table()
    if outfile is not None:
        tbl.write(outfile)
    return tbl


def get_center(filenames, box=200, outfile=None, method=1):
    '''Measure the center of disk

    filenames : array_like(str)
      Names of input images
    box : number, optional
      Size of box within which the center is measured
    outfile : str, optional
      The name of output file
    method : [1,2], optional
      Method used to measure the center. 1 - center of mass, 2 - center
      of figure

    Return : array(float), size 2xN, where N is the length of `filenames`
      The `(y, x)` centers
    '''
    cts = []
    fname = []
    for f in filenames:
        fname.append(path.basename(f))
        print(fname[-1])
        im = np.squeeze(utils.readfits(f,verbose=False))
        cts.append(utils.centroid(im, method=method, box=box))
    cts = np.array(cts)
    if outfile is not None:
        table.Table([fname, cts[:,1], cts[:,0]],names='File xc yc'.split()).write(outfile,overwrite=True)
    return cts


def background(filenames, edge=100, box=200, outfile=None):
    '''Measure the background levels of input images.  Background is measured
    as the average of resistant mean in four boxes near the four corners of
    the image.

    filenames : array_like(str)
      Names of input images
    edge : number, optional
      Width of edge to avoid when measure background
    box : number, optional
      The width of box to measure background
    outfile : str, optional
      Name of output file

    Returns: array of the background level of input images
    '''
    bgs = []
    fnames = []
    for f in filenames:
        fnames.append(path.basename(f))
        im = np.squeeze(readfits(f, verbose=False))
        sz = im.shape
        b1 = resmean(im[edge:edge+box,edge:edge+box])
        b2 = resmean(im[edge:edge+box,sz[1]-edge-box:sz[1]-edge])
        b3 = resmean(im[sz[0]-edge-box:sz[0]-edge:,edge:edge+box])
        b4 = resmean(im[sz[0]-edge-box:sz[0]-edge,sz[1]-edge-box:sz[1]-edge])
        bg = (b1+b2+b3+b4)/4.
        bgs.append([bg,b1,b2,b3,b4])
        #print('{0}: {1}'.format(fnames[-1],np.array2string(np.array(bgs[-1]),precision=3)))
        with printoptions(precision=2):
            print('{0}: {1}'.format(fnames[-1], np.array(bgs[-1])))
    bgs = np.array(bgs).T
    if outfile is not None:
        table.Table([fnames, bgs[0], bgs[1], bgs[2], bgs[3], bgs[4]], names='File Background bg1 bg2 bg3 bg4'.split()).write(outfile, overwrite=True)
    return bgs[0]


def photometry(filenames, centers, rapt=100, outfile=None):
    '''Measure integrated photometry

    filenames : array_like(str)
      Names of input images
    centers : array_like, 2xN
      Centers of images, where N is the length of `filenames`
    rapt : number of array_like, optional
      The radius of circular aperture
    outfile : str, optional
      The name of output file

    Return: array, the total flux in Jy
    '''
    import photutils as phot
    nfs = len(filenames)
    if not is_iterable(rapt):
        rapt = np.repeat(rapt,nfs)
    flux = []
    bms = []
    fname = []
    for f,c,r in zip(filenames,centers,rapt):
        fname.append(path.basename(f))
        print(fname[-1])
        im,hdr = readfits(f,verbose=False,header=True)
        im = np.squeeze(im)
        sz = im.shape
        bm = np.pi*hdr['BMAJ']*hdr['BMIN']*3600**2/4  # beam area in arcsec**2
        bms.append(bm)
        apt = phot.CircularAperture(c,r)
        ftot = apphot(im, apt)['aperture_sum'][0]
        ftot *= (hdr['cdelt1']*3600)**2/bm
        flux.append(ftot)

    flux = np.array(flux)
    bms = np.array(bms)

    fltbl = table.Table([fname,bms,flux],names='File BeamArea Flux'.split())
    if outfile is not None:
        fltbl.write(outfile,overwrite=True)

    return flux


def brightness_temperature(flux, freq, dist):
    '''Calculate brightness temperature

    flux : array or Quantity
      Input flux.  If array, then in units of Jy
    freq : array or Quantity
      Frequency of flux.  If array, then in units of Hz
    dist : array or Quantity
      Distance to Ceres.  If array, then in units of au

    Return : brightness temperature in K
    '''
    if not isinstance(flux, units.Quantity):
        flux = flux*units.Jy
    if not isinstance(freq, units.Quantity):
        freq = freq*units.Hz
    if not isinstance(dist, units.Quantity):
        dist = dist*units.au
    area = np.pi*(Ceres.r/dist).to('arcsec',equivalencies=units.dimensionless_angles())**2
    equiv = units.brightness_temperature(area, freq)
    return flux.to(units.K,equivalencies=equiv)


def imdisp(filename, ds9=None, **kwargs):
    if ds9 is None:
        ds9=saoimage.getds9()
    im,hdr = readfits(filename,verbose=False,header=True)
    im = np.squeeze(im)
    bmaj = abs(hdr['BMAJ']/(2*hdr['CDELT1']))
    bmin = abs(hdr['BMIN']/(2*hdr['CDELT1']))
    bpa = hdr['BPA']+90
    yc,xc = centroid(im, method=1, box=200)
    beam = saoimage.EllipseRegion(xc+70,yc-50,bmaj,bmin,bpa)
    width = kwargs.pop('width',None)
    if width is not None:
        beam.specs['width'] = width
    color = kwargs.pop('color',None)
    if color is not None:
        beam.specs['color'] = color
    ds9.imdisp(filename)
    beam.show(ds9)


def project(metadata, rc=(Ceres.ra.value, Ceres.rb.value, Ceres.rc.value), saveto=None):
    """Project images to lat-lon projection
    """
    # load image
    fname = path.join(metadata.path, metadata.name)
    im = utils.readfits(fname, verbose=False)
    im = np.squeeze(im)
    im /= np.pi*metadata.bmaj*metadata.bmin*1e-6/4.  # in Jy/arcsec**2

    # project to (lon, lat)
    lat,lon = utils.makenxy(-90,90,91,0,358,180)
    vpt = utils.Vector(1., metadata.SOLon, metadata.SOLat, type='geo', deg=True)
    pxlscl = metadata.Range*1.496e8*abs(metadata.xscl)/206265000.   # in km
    x,y = utils.lonlat2xy(lon, lat, rc, vpt, pa=metadata.PolePA, center=(metadata.yc,metadata.xc), pxlscl=pxlscl)
    w = np.isfinite(x) & np.isfinite(y)
    b = np.zeros_like(x)
    b[w] = im[np.round(y[w]).astype(int),np.round(x[w]).astype(int)]

    # calculate local solar time
    lst = ((lon-metadata.SSLon)/15+12) % 24

    # calculate emission angle
    emi = utils.Vector(np.ones_like(lon),lon,lat,type='geo',deg=True).vsep(utils.Vector(1,metadata.SOLon,metadata.SOLat,type='geo',deg=True))

    # save projection
    if saveto is not None:
        utils.writefits(saveto, b, overwrite=True)
        utils.writefits(saveto, lst ,name='LST', append=True)
        utils.writefits(saveto, emi, name='EMI', append=True)

    return b, lst, emi



class snell(object):

    def __init__(self, n1, n2=1., deg=True):
        self.n1 = n1
        self.n2 = n2
        self.deg = True

    def angle1(self, angle2):
        if self.deg:
            angle2 = np.deg2rad(angle2)
        a1 = np.arcsin(self.n2/self.n1 * np.sin(angle2))
        if self.deg:
            a1 = np.rad2deg(a1)
        return a1

    def angle2(self, angle1):
        if self.deg:
            angle1 = np.deg2rad(angle1)
        a2 = np.arcsin(self.n1/self.n2*np.sin(angle1))
        if self.deg:
            a2 = np.rad2deg(a2)
        return a2


# absorption length
absorption_length = lambda n, loss_tangent, wavelength=1.: wavelength/(4*np.pi*n)*(2./((1+loss_tangent*loss_tangent)**0.5-1))**0.5

# absorption length
absorption_coefficient = lambda n, loss_tangent, wavelength=1.: 1./absorption_length(n, loss_tangent, wavelength)


class surface(object):

    def __init__(self, T, n, loss_tangent=None, absorption_length=None, wavelength=1., emissivity=1.):
        self.T = T  # temperature profile where T(z) is the temperature at z
        self.n = n  # refractive index
        self.emissivity = emissivity
        self.loss_tangent = loss_tangent  # loss tangent
        if absorption_length is not None:
            # set loss tangent through absorption length and override loss_tangent parameter
            self.loss_tangent = ((2*(wavelength/(4*np.pi*self.n*absorption_length))**2+1)**2-1)**0.5
        if self.loss_tangent is None:
            raise ValueError('either `loss_tangent` or `absorption_length` has to be specified')

    def Tb(self, emi, wavelength, epsrel=1e-4):
        '''Calculate brightness temperature with subsurface emission
        accounted for'''
        if hasattr(emi,'__iter__'):
            emi = np.asanyarray(emi)
            results = np.zeros_like(emi).flatten()
            emi_flat = emi.flatten()
            for i in range(len(results)):
                results[i] = self.Tb(emi_flat[i], wavelength=wavelength, epsrel=epsrel)
            return results.reshape(emi.shape)

        s = snell(n)
        inc = s.angle1(emi)
        coef = 1/self.absorption_length(wavelength)   # absorption coefficient
        sec_i = 1./np.cos(np.deg2rad(inc))
        intfunc = lambda z: self.T(z) * np.exp(-coef*sec_i*z)
        from scipy.integrate import quad
        integral = quad(intfunc, 0, np.inf, epsrel=epsrel)[0]
        return self.emissivity*sec_i*coef * integral

    def absorption_length(self, wavelength=1.):
        '''Electrical skin depth, or absorption length
        If wavelength is not specified, then it returns Le in unit of wavelength'''
        return absorption_length(self.n, self.loss_tangent, wavelength)

