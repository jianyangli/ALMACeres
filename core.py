"""Main ALMA Ceres data processing module
"""

import json
from os import path
import numpy as np
from astropy import units, table, constants, time
from astropy.io import fits
import spiceypy as spice
from . import utils
from . import saoimage
from . import vector


# define thermal inertia units
tiu = units.def_unit('tiu', units.Unit('J/(m2 K s(0.5))'))
units.add_enabled_units(tiu)


# Ceres constants
class Ceres(object):
    """Basic properties of Ceres
    """
    # body shape and pole based on DLR RC3 shape model
    ra = 482.64 * units.km
    rb = 480.60 * units.km
    rc = 445.57 * units.km
    r = np.sqrt((ra+rb)*rc/2)
    from astropy.coordinates import SkyCoord
    pole = SkyCoord(ra=291.41*units.deg, dec=66.79*units.deg)
    GM = 62.68 * units.Unit('km3/s2')
    M = (GM/constants.G).decompose()

    # Thermal and scattering parameters, from Chamberlain et al. (2009)
    rho = 1240. * units.Unit('kg/m3')  # surface material density, Mitchell et al. (1996)
    Kr = 1.87**(rho.to('g/cm3').value)  # real dielectric component, formula from Ostro et al. 1999
    n = np.sqrt(Kr)   # refractive index
    loss_tangent = 10**(0.44*rho.to('g/cm3').value-2.943)   # loss tangent, formula from Ostro et al. 1999
    emissivity = 0.9    # emissivity, typical value
    I = 15.*tiu   # thermal inertia, Spencer 1990
    c = 750.*units.J/units.K   # specific heat, typcial rock value

    def xsection(self, dist):
        if not isinstance(dist, units.Quantity):
            dist = dist*units.au
            q = False
        else:
            q = True
        equiv = units.dimensionless_angles()
        xsec = np.pi*((Ceres.ra+Ceres.rb)/(2*dist)).to('arcsec',equivalencies=equiv)*(Ceres.rc/dist).to('arcsec', equivalencies=equiv)
        if not q:
            xsec = xsec.to('arcsec2').value
        return xsec


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
        im = np.squeeze(utils.readfits(f, verbose=False))
        sz = im.shape
        b1 = utils.resmean(im[edge:edge+box,edge:edge+box])
        b2 = utils.resmean(im[edge:edge+box,sz[1]-edge-box:sz[1]-edge])
        b3 = utils.resmean(im[sz[0]-edge-box:sz[0]-edge:,edge:edge+box])
        b4 = utils.resmean(im[sz[0]-edge-box:sz[0]-edge,sz[1]-edge-box:sz[1]-edge])
        bg = (b1+b2+b3+b4)/4.
        bgs.append([bg,b1,b2,b3,b4])
        #print('{0}: {1}'.format(fnames[-1],np.array2string(np.array(bgs[-1]),precision=3)))
        with utils.printoptions(precision=2):
            print('{0}: {1}'.format(fnames[-1], np.array(bgs[-1])))
    bgs = np.array(bgs).T
    if outfile is not None:
        table.Table([fnames, bgs[0], bgs[1], bgs[2], bgs[3], bgs[4]], names='File Background bg1 bg2 bg3 bg4'.split()).write(outfile, overwrite=True)
    return bgs[0]


class Beam():
    def __init__(self, major, minor):
        """
        Define beam size

        major, minor: number or astropy Quantity, major and minor axes.
            Default unit arcsec if not Quantity
        """
        if not isinstance(major, units.Quantity):
            major = major * units.arcsec
        self.major = major
        if not isinstance(minor, units.Quantity):
            minor = minor * units.arcsec
        self.minor = minor

    @property
    def area(self):
        # beam area: pi * bmaj * bmin / (4 * log(2))
        return np.pi*self.major*self.minor/2.772589


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
    if not utils.is_iterable(rapt):
        rapt = np.repeat(rapt,nfs)
    flux = []
    bms = []
    fname = []
    for f,c,r in zip(filenames,centers,rapt):
        fname.append(path.basename(f))
        print(fname[-1])
        im,hdr = utils.readfits(f,verbose=False,header=True)
        im = np.squeeze(im)
        sz = im.shape
        bm = Beam(hdr['BMAJ']*units.deg,hdr['BMIN']*units.deg).area.to('arcsec2').value
        bms.append(bm)
        apt = phot.CircularAperture(c,r)
        ftot = utils.apphot(im, apt)['aperture_sum'][0]
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
    area = Ceres().xsection(dist)
    equiv = units.brightness_temperature(area, freq)
    return flux.to(units.K,equivalencies=equiv)


def imdisp(filename, ds9=None, **kwargs):
    if utils.is_iterable(filename):
        for f in filename:
            imdisp(f, ds9=ds9, **kwargs)
    else:
        if ds9 is None:
            ds9=saoimage.getds9()
        im,hdr = utils.readfits(filename,verbose=False,header=True)
        im = np.squeeze(im)
        bmaj = abs(hdr['BMAJ']/(2*hdr['CDELT1']))
        bmin = abs(hdr['BMIN']/(2*hdr['CDELT1']))
        bpa = hdr['BPA']+90
        yc,xc = utils.centroid(im, method=1, box=200)
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
    vpt = vector.Vector(1., metadata.SOLon, metadata.SOLat, type='geo', deg=True)
    pxlscl = metadata.Range*1.496e8*abs(metadata.xscl)/206265000.   # in km
    x,y = vector.lonlat2xy(lon, lat, rc, vpt, pa=metadata.PolePA, center=(metadata.yc,metadata.xc), pxlscl=pxlscl)
    w = np.isfinite(x) & np.isfinite(y)
    b = np.zeros_like(x)
    b[w] = im[np.round(y[w]).astype(int),np.round(x[w]).astype(int)]

    # calculate local solar time
    lst = ((lon-metadata.SSLon)/15+12) % 24

    # calculate emission angle
    emi = vector.Vector(np.ones_like(lon),lon,lat,type='geo',deg=True).vsep(vector.Vector(1,metadata.SOLon,metadata.SOLat,type='geo',deg=True))

    # save projection
    if saveto is not None:
        utils.writefits(saveto, b, overwrite=True)
        utils.writefits(saveto, lst ,name='LST', append=True)
        utils.writefits(saveto, emi, name='EMI', append=True)

    return b, lst, emi


class Snell(object):
    """
    Implementation of Snell's law of reflection.  In this model, radiation is
    transmitted from the first media to the second media.  This model
    calculates the angles of refraction.
    """

    def __init__(self, n1, n2=1.):
        """
        n1: number, refractive index of the first media
        n2: optional, number, refractive index of the second media
        """
        self.n1 = n1
        self.n2 = n2


    @staticmethod
    def refraction_angles(angle, n1, n2):
        """
        Calculates the refractive angle in the second media from that in the
        first media.

        angle: number, astropy Quantity, or array-like number or Quantity,
            the angle(s) in the second media.  If not Quantity, then in
            degrees by default.
        n1: number, refractive index of the first (incident) media
        n2: number, refractive index of the second (emission) media
        """
        sinangle = n1/n2 * np.sin(np.deg2rad(angle))
        if sinangle > 1:
            return np.nan
        a = np.arcsin(sinangle)
        if not isinstance(angle, units.Quantity):
            a = np.rad2deg(a)
        return a


    def angle1(self, angle2):
        """
        Calculates the refractive angle in the second media from that in the
        first media.

        angle2: number, astropy Quantity, or array-like number or Quantity,
            the angle(s) in the second media.  If not Quantity, then in
            degrees by default.
        """
        return Snell.refraction_angles(angle2, self.n2, self.n1)


    def angle2(self, angle1):
        """
        Calculates the refractive angle in the second media from that in the
        first media.

        angle2: number, astropy Quantity, or array-like number or Quantity,
            the angle(s) in the second media.  If not Quantity, then in
            degrees by default.
        """
        return Snell.refraction_angles(angle1, self.n1, self.n2)


    @property
    def critical_angle(self):
        """Critial angle of reflection"""
        n1 = self.n1
        n2 = self.n2
        if n1 > n2:
            n1, n2 = n2, n1
        return np.rad2deg(np.arcsin(n1/n2))


# absorption length
absorption_length = lambda n, loss_tangent, wavelength=1.: wavelength/(4*np.pi*n)*(2./((1+loss_tangent*loss_tangent)**0.5-1))**0.5

# absorption length
absorption_coefficient = lambda n, loss_tangent, wavelength=1.: 1./absorption_length(n, loss_tangent, wavelength)

# loss tangent
loss_tangent = lambda n, abs_len, wavelength: ((2*(wavelength/(4*np.pi*n*abs_len))**2+1)**2-1)**0.5


class Layer(object):
    """Layer class for calculating propagation of subsurface thermal emission

    Based on the models in Keihm & Langseth (1975), Icarus 24, 211-230
    """

    def __init__(self, n, loss_tangent, depth=np.inf, emissivity=1., profile=None):
        """
        n: number, refractive index of this layer
        loss_tangent: number, loss tengent
        depth: optional, number, depth of the layer in the same unit as `z`
            (see below for the description of `profile`)
        emissivity: optional, number, emissivity
        profile: optional, callable object, where `profile(z)` returns the
            physical quantity of the layer at depth `z`.  Most commonly it is a
            temperature profile, or a thermal emission profile, etc.
        """
        self.n = n
        self.depth = depth
        self.loss_tangent = loss_tangent
        self.emissivity = emissivity
        self.profile = profile


    def absorption_length(self, wavelength=1.):
        """
        Calculates absorption length

        n: number, refractive index
        loss_tangent: number, loss tangent
        wavelength: optional, number or astropy Quantity or array_like number
            or Quantity, wavelength of observations
        """
        return 1./self.absorption_coefficient(wavelength=wavelength)


    def absorption_coefficient(self, wavelength=1.):
        """
        Calculates absorption coefficient

        n: number, refractive index
        loss_tangent: number, loss tangent
        wavelength: optional, number or astropy Quantity or array_like number
            or Quantity, wavelength of observations
        """
        return (4*np.pi*self.n)/wavelength*np.power(0.5*(np.power(1+self.loss_tangent*self.loss_tangent,0.5)-1),0.5)


class Surface(object):
    """
    Surface class that contain subsurface layers and calculates the observables
    from the surface, such as brightness temperature or thermal emission, etc.

    Based on the models in Keihm & Langseth (1975), Icarus 24, 211-230
    """

    def __init__(self, layers):
        """
        layers: SubsurfaceLayer class object or array_like of it, the
            subsurface layers.  If array-like, then the layers are ordered
            from the top surface downward.
        """
        if not hasattr(layers, '__iter__'):
            self.layers = [layers]
        else:
            self.layers = layers
        self.depth = 0
        self._check_layer_depth()
        self.depth = np.sum([l.depth for l in self.layers])
        self.n_layers = len(self.layers)


    def _check_layer_depth(self):
        for i,l in enumerate(self.layers[:-1]):
            if l.depth == np.inf:
                raise ValueError(f'only the deepest layer can have infinite depth.  the depth of the {i}th layer cannot be infinity')


    def _check_layer_profile(self):
        for i,l in enumerate(self.layers):
            if l.profile is None:
                raise ValueError(f'the {i}th layer does not have a quantity profile defined')
            if not hasattr(l.profile, '__call__'):
                raise ValueError(f'the {i}th layer does not have a valid quantity profile defined')


    def emission(self, emi_ang, wavelength, epsrel=1e-4, debug=False):
        """Calculates the quantity at the surface with subsurface emission
        propagated and accounted for.

        emi_ang: number or astropy Quantity, emission angle.  If not Quantity,
            then angle is in degrees
        wavelength: wavelength to calculate, same unit as the length quantities
            in `Surface.layers` class objects
        epsrel: optional, relative error to tolerance in numerical
            integration.  See `scipy.integrate.quad`.
        """
        if hasattr(emi_ang,'__iter__'):
            raise ValueError('`emi_ang` has to be a scaler')
        self._check_layer_depth()
        self._check_layer_profile()

        L = 0
        n0 = 1.
        m = 0.
        if debug:
            D = 0
            prof = {'t': [], 'intprofile': [], 'zzz': [], 'L0': []}
        for i,l in enumerate(self.layers):
            # integrate in layer `l`
            inc = Snell(l.n, n0).angle1(emi_ang)
            coef = l.absorption_coefficient(wavelength)
            cos_i = np.cos(np.deg2rad(inc))
            if debug:
                print(f'cos(i) = {cos_i}, coef = {coef}')
            intfunc = lambda z: l.profile(z) * np.exp(-coef*z/cos_i-L)
            if l.depth == np.inf:
                zz = np.linspace(0,1000,100)
            else:
                zz = np.linspace(0,l.depth,100)
            prof['t'].append(l.profile(zz))
            prof['intprofile'].append(intfunc(zz))
            prof['zzz'].append(zz+D)
            prof['L0'].append(L)
            from scipy.integrate import quad
            integral = quad(intfunc, 0, l.depth, epsrel=epsrel)[0]
            m += l.emissivity*coef*integral/cos_i
            # prepare for the next layer
            D += l.depth
            L += l.depth/cos_i*coef
            emi_ang = np.arccos(cos_i)
            n0 = l.n

        if debug:
            return m, prof
        else:
            return m
