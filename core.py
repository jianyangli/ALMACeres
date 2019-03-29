"""Main ALMA Ceres data processing module
"""

import warnings
import json
from os import path
import numpy as np
from astropy import table, time
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
import spiceypy as spice
from . import utils
from . import saoimage
from . import vector


# define thermal inertia units
tiu = u.def_unit('tiu', u.Unit('J/(m2 K s(0.5))'))
u.add_enabled_units(tiu)

solar_constant = (const.L_sun/(4*np.pi*u.au**2)).to('W/m2')

# absorption length
absorption_length = lambda n, loss_tangent, wavelength=1.: wavelength/(4*np.pi*n)*(2./((1+loss_tangent*loss_tangent)**0.5-1))**0.5

# absorption length
absorption_coefficient = lambda n, loss_tangent, wavelength=1.: 1./absorption_length(n, loss_tangent, wavelength)

# loss tangent
loss_tangent = lambda n, abs_len, wavelength: ((2*(wavelength/(4*np.pi*n*abs_len))**2+1)**2-1)**0.5


class TriaxialShape(object):

    @u.quantity_input
    def __init__(self, ra: u.km, rb: u.km, rc: u.km):
        self.ra = ra
        self.rb = rb
        self.rc = rc
        self.r_volume_equivalent = (ra*rb*rc)**(1/3)
        self.r_area_equivalent = np.sqrt((ra+rb)*rc/2)

    @property
    def surface_area(self):
        p = 1.6075
        ap = self.ra**p
        bp = self.rb**p
        cp = self.rc**p
        return 4*np.pi*((ap*bp+ap*cp+bp*cp)/3)**(1/p)

    @property
    def volume(self):
        return 4/3*np.pi*self.ra*self.rb*self.rc


class Body(object):
    pass


# Ceres constants
class Ceres(Body):
    """Basic properties of Ceres
    """

    def __init__(self):
        # body shape and pole based on DLR RC3 shape model
        self.shape = TriaxialShape(482.64*u.km, 480.64*u.km, 445.57*u.km)
        from astropy.coordinates import SkyCoord
        self.pole = SkyCoord(ra=291.41*u.deg, dec=66.79*u.deg)
        self.GM = 62.68 * u.Unit('km3/s2')

        # Thermal and scattering parameters, from Chamberlain et al. (2009)
        self.surface_density = 1240. * u.Unit('kg/m3')  # surface material density, Mitchell et al. (1996)
        self.Kr = 1.87**(self.surface_density.to('g/cm3').value)  # real dielectric component, formula from Ostro et al. 1999
        self.loss_tangent = 10**(0.44*self.surface_density.to('g/cm3').value-2.943)   # loss tangent, formula from Ostro et al. 1999
        self.emissivity = 0.9    # emissivity, typical value
        self.thermal_inertia = 15.*tiu   # thermal inertia, Spencer 1990
        self.specific_heat = 750.*u.J/(u.K*u.kg)   # specific heat, typcial rock value
        self.rotational_period = 9.075 * u.hour
        self.Bond_albedo = 0.03

    @property
    def M(self):
        return (self.GM/const.G).to('kg')

    @property
    def bulk_density(self):
        return (self.M/self.shape.volume).to('kg/m3')

    @property
    def refractive_index(self):
        return np.sqrt(self.Kr)

    def xsection(self, dist):
        if not isinstance(dist, u.Quantity):
            dist = dist*u.au
            q = False
        else:
            q = True
        equiv = u.dimensionless_angles()
        xsec = np.pi*((Ceres.ra+Ceres.rb)/(2*dist)).to('arcsec',equivalencies=equiv)*(Ceres.rc/dist).to('arcsec', equivalencies=equiv)
        if not q:
            xsec = xsec.to('arcsec2').value
        return xsec


ceres = Ceres()


class ALMACeresImageMetaData(utils.MetaData):
    """Meta data class for Ceres ALMA images
    """
    def collect(self, datafile, spicekernel=None, **kwargs):
        super().collect(datafile)
        hdr = fits.open(datafile)[0].header
        self.bmaj = hdr['bmaj']*3600_000.
        self.bmin = hdr['bmin']*3600_000.
        self.bpa = hdr['bpa']
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
        if not isinstance(major, u.Quantity):
            major = major * u.arcsec
        self.major = major
        if not isinstance(minor, u.Quantity):
            minor = minor * u.arcsec
        self.minor = minor

    @property
    def area(self):
        # returns a Quantity, beam area: pi * bmaj * bmin / (4 * log(2))
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
        bm = Beam(hdr['BMAJ']*u.deg,hdr['BMIN']*u.deg).area.to('arcsec2').value
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
    if not isinstance(flux, u.Quantity):
        flux = flux*u.Jy
    if not isinstance(freq, u.Quantity):
        freq = freq*u.Hz
    if not isinstance(dist, u.Quantity):
        dist = dist*u.au
    area = Ceres().xsection(dist)
    equiv = u.brightness_temperature(area, freq)
    return flux.to(u.K,equivalencies=equiv)


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


def project(metadata, rc=(ceres.shape.ra.value, ceres.shape.rb.value, ceres.shape.rc.value), saveto=None):
    """Project images to lat-lon projection
    """
    # load image
    fname = path.join(metadata.path, metadata.name)
    im = utils.readfits(fname, verbose=False)
    im = np.squeeze(im)
    im /= Beam(metadata.bmaj*u.mas,metadata.bmin*u.mas).area.to('arcsec2').value  # in Jy/arcsec**2

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
        if not isinstance(angle, u.Quantity):
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
            if debug:
                prof['t'].append(l.profile(zz))
                prof['intprofile'].append(intfunc(zz))
                prof['zzz'].append(zz+D)
                prof['L0'].append(L)
                D += l.depth
            from scipy.integrate import quad
            integral = quad(intfunc, 0, l.depth, epsrel=epsrel)[0]
            m += l.emissivity*coef*integral/cos_i
            # prepare for the next layer
            L += l.depth/cos_i*coef
            emi_ang = np.arccos(cos_i)
            n0 = l.n

        if debug:
            return m, prof
        else:
            return m


def _shift_1d_array(a, i):
    out = np.empty_like(a)
    out[:i] = a[-i:]
    out[i:] = a[:-i]
    return out


class Thermal():

    """Thermophysical model class.  This class collects the thermal parameters
    to calculate various thermal parameters and 1d temperature model based on
    thermophysical model.

    For the simplest dimensionless thermal model, the only parameter needed is
    the dimensionless "thermal parameter", or the "big Theta".  In this case,
    the temperature model, time, and depth are all dimensionless.

    If sufficient parameters are supplied, then the temperature model can be
    dimensional with the corresponding physical units.

    The computed `.temperature_model` attributeis a 2-D floating point or
    `astropy.units.Quantity` array of shape (nt, ceil(z1/dz)+1).  Each
    line represents the temperature profile with respect to depth at a
    particular time; and each column represents the temperature profile as
    a function of time at a certain depth.  The temperature is either
    dimensionless relative to the sub-solar temperature or in physical
    unit in equilibrium thermal model.

    Examples
    --------

    # initialize a dimentionless model and generate temperature distribution
    >>> from ALMACeres import Thermal
    >>> t = Thermal(Theta=1.)
    >>> t.thermal_model()
    >>> print(type(t.temperature_model))
    numpy.ndarray
    >>> print(t.temperature_model.shape)
    (360, 101)

    # initialize the model with Ceres at 2.77 au, and generate temperature
    # distribution
    >>> import astropy.units as u
    >>> from ALMACeres import Theta, Ceres
    >>> ceres = Ceres()
    >>> t = Thermal(body=ceres, rh=2.77*u.au)
    >>> t.thermal_model()
    >>> print(type(t.temperature_model))
    <class 'astropy.units.quantity.Quantity'>
    >>> print(t.temperature_model.unit)
    (360, 101)

    Notes
    -----
    This program follows the numerical approach discussed in Spencer et al.
    (1989, Icarus 78, 337-354).

    The calculation starts from the specified initial condition, step in time,
    and iteration for rotations until the temperature profile with respect to
    both depth and time stablizes or the maximum number of iteration is
    reached.  The stablization is determined by the RMS change of temperature
    profile from one rotation to the next.  The RMS difference is defined as
        RMS = np.sqrt(np.mean(difference_temperature**2))

    The step sizes in both time and depth affect the stablization of numerical
    solution.  The default number of steps of 360 in one rotation and default
    step size of 0.5 in depth work well with big_theta=1.  For other values of
    big_theta, these parameters may need to be adjusted accordingly.
    """

    @u.quantity_input(equivalencies=u.temperature())
    def __init__(self, body=None,
            conductivity: 'W/(m K)'=None, density: 'kg/m3'=None,
            specific_heat: 'J/(kg K)'=None, thermal_inertia: tiu=None,
            Theta=None, Omega: '1/s'=None, Period: 's'=None, Tss: 'K'=None,
            Ab=None, emissivity=None, rh: 'au'=None):
        if not isinstance(body, Body):
            self.body = None
            self._conductivity = conductivity
            self._density = density
            self._specific_heat = specific_heat
            self._thermal_inertia = thermal_inertia
            self._Theta = Theta
            self._Omega = Omega
            self._Period = Period
            self._Tss = Tss
            self._Ab = Ab
            self.emissivity = emissivity
            self._rh = rh
        else:
            self._body = body
            self._conductivity = getattr(body, 'thermal_conductivity', None)
            self._density = getattr(body, 'surface_density', None)
            self._specific_heat = getattr(body, 'specific_heat', None)
            self._thermal_inertia = getattr(body, 'thermal_inertia', None)
            self._Theta = Theta
            self._Omega = Omega
            self._Period = getattr(body, 'rotational_period', None)
            self._Tss = Tss
            self._Ab = getattr(body, 'Bond_albedo', None)
            self.emissivity = getattr(body, 'emissivity', None)
            self._rh = rh

        if self._conductivity is not None:
            self._conductivity = self._conductivity.to('W/(m K)')
        if self._density is not None:
            self._density = self._density.to('kg/m3')
        if self._specific_heat is not None:
            self._specific_heat = self._specific_heat.to('J/(kg K)')
        if self._thermal_inertia is not None:
            self._thermal_inertia = self._thermal_inertia.to(tiu)
        if self._Theta is not None:
            self._Theta = self._Theta * u.dimensionless_unscaled
        if self._Omega is not None:
            self._Omega = self._Omega.to('1/s')
        if self._Period is not None:
            self._Period = self._Period.to('s')
        if self._Tss is not None:
            self._Tss = self._Tss.to('K', u.temperature())
        if self._Ab is not None:
            self._Ab = self._Ab * u.dimensionless_unscaled
        if self.emissivity is not None:
            self.emissivity = self.emissivity * u.dimensionless_unscaled
        if self._rh is not None:
            self._rh = self._rh.to('au')

    @property
    def conductivity(self):
        if self._conductivity is None:
            try:
                self._conductivity = (self._thermal_inertia**2/(self._density*self._specific_heat)).to('W/(m K)')
            except TypeError:
                pass
        return self._conductivity
    @conductivity.setter
    @u.quantity_input(var=['W/(m K)', None])
    def conductivity(self, var):
        if var is None:
            self._conductivity = None
        else:
            self._conductivity = var.to('W/(m K)')

    @property
    def density(self):
        if self._density is None:
            try:
                self._density = (self._thermal_inertia**2/(self._conductivity*self._specific_heat)).to('kg/m3')
            except TypeError:
                pass
        return self._density
    @density.setter
    @u.quantity_input(var=['kg/m3', None])
    def density(self, var):
        if var is None:
            self._density = None
        else:
            self._density = var.to('kg/m3')

    @property
    def specific_heat(self):
        if self._specific_heat is None:
            try:
                self._specific_heat = (self._thermal_inertia**2/(self._conductivity*self._density)).to('J/(kg K)')
            except TypeError:
                pass
        return self._specific_heat
    @specific_heat.setter
    @u.quantity_input(var=['J/(kg K)', None])
    def specific_heat(self, var):
        if var is None:
            self._specific_heat = None
        else:
            self._specific_heat = var.to('J/(kg K)')

    @property
    def thermal_inertia(self):
        if self._thermal_inertia is None:
            try:
                self._thermal_inertia = np.sqrt(self._conductivity*self._density*self._specific_heat).to(tiu)
            except TypeError:
                pass
        return self._thermal_inertia
    @thermal_inertia.setter
    @u.quantity_input(var=[tiu, None])
    def thermal_inertia(self, var):
        if var is None:
            self._thermal_inertia = None
        else:
            self._thermal_inertia = var.to(tiu)

    @property
    def Theta(self):
        if self._Theta is None:
            if self._thermal_inertia is None:
                thermal_inertia = self.thermal_inertia
            if self._Omega is None:
                Omega = self.Omega
            if self._Tss is None:
                Tss = self.Tss
            try:
                self._Theta = (self._thermal_inertia*np.sqrt(self._Omega)/(self.emissivity*const.sigma_sb*self._Tss**3)).decompose()
            except (TypeError, AttributeError):
                pass
        return self._Theta
    @Theta.setter
    def Theta(self, var):
        if var is None:
            self._Theta = var
        else:
            self._Theta = var * u.dimensionless_unscaled

    @property
    def Omega(self):
        if self._Omega is None:
            try:
                self._Omega = (2*np.pi/self._Period).to('1/s')
            except TypeError:
                pass
        return self._Omega
    @Omega.setter
    @u.quantity_input(var=['1/s', None])
    def Omega(self, var):
        if var is None:
            self._Omega = None
        else:
            self._Omega = var.to('1/s')

    @property
    def Period(self):
        if self._Period is None:
            try:
                self._Period = (2*np.pi/self._Omega).to('s')
            except TypeError:
                pass
        return self._Period
    @Period.setter
    @u.quantity_input(var=['s', None])
    def Period(self, var):
        if var is None:
            self._Period = None
        else:
            self._Period = var.to('s')

    @property
    def Tss(self):
        if self._Tss is None:
            try:
                self._Tss = ((1-self._Ab)*solar_constant/self._rh.to('au').value**2/(self.emissivity*const.sigma_sb))**0.25
            except (TypeError, AttributeError):
                pass
        return self._Tss
    @Tss.setter
    @u.quantity_input(var='K')
    def Tss(self, var):
        if var is None:
            self._Tss = None
        else:
            self._Tss = var.to('K')

    @property
    def Ab(self):
        return self._Ab
    @Ab.setter
    def Ab(self, var):
        if var is None:
            self._Ab = None
        else:
            self._Ab = var * u.dimensionless_unscaled

    @property
    def skin_depth(self):
        try:
            if self._Omega is None:
                Omega = self.Omega
            if self._conductivity is None:
                conductivity = self.conductivity
            if self._density is None:
                density = self.density
            if self._specific_heat is None:
                specific_heat = self.specific_heat
            skin_depth = np.sqrt(self._conductivity/(self._density*self._specific_heat*self._Omega)).to('m')
        except TypeError:
            skin_depth = None
        return skin_depth

    @property
    def rh(self):
        return self._rh
    @rh.setter
    @u.quantity_input(var='au')
    def rh(self, var):
        if var is None:
            self._rh = None
        else:
            self._rh = var.to('au')

    def thermal_model(self, nt=360, dz=0.5, z1=50, init=None, maxiter=10000,
            tol=1e-5, inc=None, cos=False, verbose=False, benchmark=False):
        """Calculate the temperature profile on and inside the surface of a
        rotating body according to thermal physical model.

        Parameters
        ----------

            nt : Number of time steps in one rotation.  An integer number.
                Default is 360 if not specified.
            dz : Step size in dimensionless depth.  Default is 0.5.
            z1 : The inside boundary of dimensionless depth.  Default is 50.
            init : Initial 2-D temperature distribution with the same
                structure as the returned array.  It has to be a 2-D array
                variable or values.  Default is 0.6*max(cos(inc)) for all
                depth and all times.
            maxiter : Maximum number of iterations (rotations).  An integer
                value or variable.  Default is 4294967295.
            tol : Tolerance of convergence.  A floating point variable or
                value.  Default is 1e-5.
            inc : An array to specify the solar incidence angle in one day.
                It has to have the same number of elements as specified by
                parameter nt.  The values of angles are either in degress, if
                keyword "cos" is not set, or the cosine of angles, if keyword
                "cos" is set.  User has to ensure that incidence angle is
                physically meaningful, i.e., for incidence angle higher than 90
                deg, user should set it to be 90.
            cos : If set, then keyward inc passes values in cos(incidence
                angle).  By default, the unit of inc is deg.
            verbose : Enable verbose mode for the program to print out
                information and progress after each iteration.
            benchmark : If set this keyword, program will run with keywords
                plot and verbose disabled, and print out the CPU time in
                seconds used for calculation.  This keyword can be used to
                benchmark the speed of calculation.

        Returns
        -------
        None.  Upon return, this method will populate two class attribute,
        `.temperature_model`, which is a 2D Quantity array stores the model
        temperature with respect to time and depth, and `.model_param`, which
        is a `dict` that stores the model parameters.

        The items in
        `.model_param` are:
        'z' : 1d Quantity array stores the depth corresponding to
            `.temperature_model`
        't' : 1d Quantity array stores the times corresponding to
            `.temperature_model`
        'insolation' : 1d Quantity array stores the solar flux at times stored
            in `.model_param['t']
        'niter' : number of iterations to converge
        """
        if self.Theta is None:
            warnings.warn('aborted: thermal parameter is unknown')
            return

        self.model_param = {}
        nz = int(np.ceil(z1/dz) + 1)  # number of depth steps
        zz = np.arange(nz) * dz  # depths
        self.model_param['z'] = zz
        dt = 2 * np.pi / nt  # number of time steps
        tt = np.arange(nt) * dt  # time
        self.model_param['t'] = tt

        if self.skin_depth is None:
            warnings.warn("skip depth cannot be calculated, depth parameter `.model_param['z']` will be dimensionless")
        else:
            self.model_param['z'] = self.model_param['z'] * self.skin_depth
        if self.Omega is None:
            warnings.warn("rotational period unknown, time parameter `.model_param['t']` will be dimensionless")
        else:
            self.model_param['t'] = self.model_param['t'] / self.Omega


        if inc is not None:
            if len(inc) == nt:
                if cos:
                    insol = inc
                else:
                    insol = np.cos(np.deg2rad(inc))
                if (insol < 0).any():
                    warnings.warn('some solar incidence angle < 90 deg')
            else:
                warnings.warn('wrong parameter `inc` ignored!')
                insol = np.clip(np.cos(tt), 0, None)
                insol = _shift_1d_array(insol, nt/2)
        else:
            insol = np.clip(np.cos(tt), 0, None)
            insol = _shift_1d_array(insol, nt//2)
        self.model_param['insolation'] = insol
        if self.rh is None:
            warnings.warn("heliocentric distance is unknown, solar insolation parameter `.model_param['insolation']` will be dimensionless")
        else:
            self.model_param['insolation'] = self.model_param['insolation'] * solar_constant / self.rh.to('au').value**2

        r = dt / (dz*dz)
        if r >= 0.5:
            warnings.warn('time step is too large, and solution may not converge')

        if verbose:
            print('Solve 1-D heat conduction equation:')
            print(f'    Thermal parameter = {self.Theta:.3f}')
            print(f'    Time steps = {nt:6d} per cycle')
            print(f'    Inner depth = {z1:.3f} thermal skin depth(s)')
            print(f'    Depth step size = {dz:.3f}')
            print(f'    dt/(dz^2) = {r:.3f}')
            print(f'    Maximum iteration = {maxiter}')
            print(f'    RMS Tolerance = {tol:.3g}')
            print()

        if init is None:
            uu = np.ones((nt, nz)) * 0.6 * insol.max()
        else:
            uu0 = np.asarray(init)
            if uu0.shape != (nt, nz):
                warnings.warn('Warning: Invalid initial condition is ignored!')
                uu = np.ones((nt, nz)) * 0.6 * insol.max()
            else:
                uu = uu0

        if benchmark:
            import time
            t0 = time.time()

        niter = 0
        rms = 1.
        while (rms > tol) and (niter < maxiter):
            uu1 = uu.copy()
            # loop through one rotation
            for i in range(nt):
                # index for i+1'th time
                next_step = (i+1) % nt
                # propagate one time step
                uu[next_step] = (1-2*r)*uu[i]+r*(_shift_1d_array(uu[i],1)+_shift_1d_array(uu[i],-1))
                # boundary conditions
                # surface
                uu[next_step,0] = uu[i,0]+2*r*(uu[i,1]-uu[i,0])-2*dt/(dz*self.Theta)*(uu[i,0]**4-insol[i])
                # inside
                uu[next_step,-1] = uu[next_step,-2]
            # increase iteration counter
            niter += 1
            # calculate RMS difference
            diff = uu1-uu
            ww = uu != 0
            rdiff = diff[ww]/uu[ww]
            # rms = np.clip(np.sqrt(np.mean(rdiff*rdiff)), None, np.sqrt(np.mean(diff*diff))*1e4)
            rms = np.sqrt((rdiff*rdiff).mean())

            # print out progress
            if verbose:
                # find the time of maximum temperature on the surface
                ww = uu[:,0].argmax()
                print(f'Iter #{niter:6d}: RMS = {rms:.3g}, Max. surf. T = {uu.max():.3f} @ LST {(tt[ww]/(2*np.pi)*24) % 24:.2f} hrs, Term. T = {uu[:,nz-1].min():.3f}')

        if benchmark:
            print(f'Time for {niter:6d} iterations: {time.time()-t0:.3f} sec')

        self.model_param['niter'] = niter
        self.temperature_model = uu
        if self.Tss is None:
            warnings.warn("subsolar temperature is unknown, temperature model `.temperature_model` will be dimensionless")
        else:
            self.temperature_model = self.temperature_model * self.Tss
        self.model_param['lst'] = self.model_param['t']/self.Period*24*u.hour

        # print out information
        if verbose:
            print()
            print(f'Total iterations: {niter:6d}')


class SphereSurfaceTemperature():

    @u.quantity_input
    def __init__(self, tpm, sunlat: u.deg=0*u.deg, dlat: u.deg=5*u.deg):
        self.tpm = tpm
        self.sunlat = sunlat
        self.dlat = dlat

    def thermal_model(self, nt=360, dz=0.5, z1=50, init=None, maxiter=10000, tol=1e-4, verbose=False):

        if self.tpm.Theta is None:
            warnings.warn('aborted: thermal parameter is unknown')
            return

        self.model_param = {}
        nz = int(np.ceil(z1/dz)+1)  # number of steps in depth
        nlat = int(180/self.dlat.to('deg').value+1)   # number of latitudinal zones
        lats =  np.linspace(-90, 90, nlat)  # latitudes of all zones
        self.model_param['latitudes'] = lats * u.deg

        # initial condition
        if init is not None:
            tt0 = np.asarray(init)
            if tt0.shape == (nt, nz, nlat):
                tt = tt0
            else:
                warnings.warn('invalid initial condition is ignored')
                tt = np.ones((nt, nz, nlat)) * 0.4
        else:
            tt = np.ones((nt, nz, nlat)) * 0.4

        # solar vector assuming solar longitude 0
        sunvec = vector.Vector(1, 0, self.sunlat.to('rad').value, type='geo')
        lons = np.arange(nt)/nt*360   # longitudes for all time steps
        lons = _shift_1d_array(lons, nt//2)

        # loop through all latitudinal zones
        niter = np.zeros(nlat, dtype=int)
        insol = []
        for i, lt in enumerate(lats):
            if verbose:
                print()
                print(f'Latitude = {lt:.2f} deg')
                print()

            # calculate solar incidence angles at all time steps
            norm = vector.Vector(np.ones_like(lons), lons, np.repeat(lt, nt), type='geo', deg=True)
            inc = np.clip(np.cos(sunvec.vsep(norm)), 0, None)

            # calculate temperature only when the sun raises above horizon
            # some time in a day
            if inc.max() > 1e-5:
                self.tpm.thermal_model(nt=nt, dz=dz ,z1=z1 ,inc=inc, cos=True, verbose=verbose ,tol=tol, init=np.squeeze(tt[...,i]), maxiter=maxiter)
                tt[...,i] = self.tpm.temperature_model.copy()
                niter[i] = self.tpm.model_param['niter']
                insol.append(self.tpm.model_param['insolation'])
            else:
                tt[...,i] = 0.
                niter[i] = 0
                insol.append(np.zeros(nt))

        isquan = [isinstance(x, u.Quantity) for x in insol]
        if np.any(isquan):
            for i in range(len(insol)):
                if not isquan[i]:
                    insol[i] = insol[i]*u.Unit('W/m2')
        insol = u.Quantity(insol)

        self.temperature_model = tt
        self.model_param['z'] = self.tpm.model_param['z'].copy()
        self.model_param['t'] = self.tpm.model_param['t'].copy()
        self.model_param['lst'] = self.tpm.model_param['lst'].copy()
        self.model_param['insolation'] = insol
        self.model_param['niter'] = niter

    def save(self, file, overwrite=False):
        """Save model temperature to file"""
        hdu = fits.PrimaryHDU(self.temperature_model.to('K').value)
        if self.tpm.skin_depth is not None:
            hdu.header['skindep'] = self.tpm.skin_depth.to('m').value
        if self.tpm.Period is not None:
            hdu.header['period'] = self.tpm.Period.to('s').value
        hdr.writeto(file, overwrite=overwrite)
        utils.writefits(file, self.model_param['t'].to('s').value, name='Time', append=True)
        utils.writefits(file, self.model_param['lst'].to('hour').value, name='LST', append=True)
        utils.writefits(file, self.model_param['z'].to('m').value, name='Depth', append=True)
        utils.writefits(file, self.model_param['latitudes'].to('deg').value, name='Lagitude', append=True)
        utils.writefits(file, self.model_param['niter'], name='niter', append=True)

