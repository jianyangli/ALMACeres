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
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
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
        self.r = u.Quantity([ra, rb, rc])

    @property
    def r_volume_equivalent(self):
        return (self.a*self.b*self.c)**(1/3)

    @property
    def r_area_equivalent(self):
        return np.sqrt((self.a+self.b)*self.c/2)

    @property
    def surface_area(self):
        p = 1.6075
        ap = self.a**p
        bp = self.b**p
        cp = self.c**p
        return 4*np.pi*((ap*bp+ap*cp+bp*cp)/3)**(1/p)

    @property
    def volume(self):
        return 4/3*np.pi*self.a*self.b*self.c

    @property
    def a(self):
        return self.r[0]

    @a.setter
    @u.quantity_input
    def a(self, value: u.km):
        self.r[0] = value

    @property
    def b(self):
        return self.r[1]

    @b.setter
    @u.quantity_input
    def b(self, value: u.km):
        self.r[1] = value

    @property
    def c(self):
        return self.r[2]

    @c.setter
    @u.quantity_input
    def c(self, value: u.km):
        self.r[2] = value

    @property
    def equatorial_cross_section(self):
        return np.pi*(self.a+self.b)*self.c/2

    @property
    def polar_cross_section(self):
        return np.pi*self.a*self.b


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
        self.n = 1.3

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
        xsec = (self.shape.equatorial_cross_section / (dist*dist)).to('arcsec2', equiv)
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


class ALMAImage(u.Quantity):
    """ALMA image class

    This is a subclass of `astropy.units.Quantity`, which subclasses
    `np.ndarray`.  In addition to all the `u.Quantity` attributes, it
    also contains a `.meta` attribute to store all the information
    related to ALMA images.  WCS can also be stored in `.meta['wcs']`
    if available.
    """
    def __new__(cls, input_array, meta=None, header=None, **kwargs):
        obj = u.Quantity(input_array, **kwargs).view(cls)
        obj.meta = meta
        obj.header = header
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None: return
        self.meta = getattr(obj, 'meta', None)
        self.header = getattr(obj, 'header', None)

    @classmethod
    def from_fits(cls, filename, wcs_kwargs={}, skycoord_kwargs={}):
        """Initialize class from FITS file

        Parameters
        ----------
        filename : str or `astropy.io.fits.HDUList`
            Input FITS file or object
        wcs_kwargs : dict, optional
            Keywords to be passed to `astropy.wcs.WCS()`
        skycoord_kwargs : dict, optional
            Keywords to be passed to `astropy.coordinates.SkyCoord()`

        Return
        ------
        `ALMAImage` object
        """
        if isinstance(filename, str):
            im = fits.open(filename)
        elif isinstance(filename, fits.HDUList):
            im = filename
        else:
            raise TypeError('Unrecogniazed input `filename`.')
        hdr = im[0].header
        data = (im[0].data * hdr['bscale'] + hdr['bzero']) * \
                u.Unit(hdr['BUNIT'])
        hdr.remove('bscale')
        hdr.remove('bzero')
        hdr.remove('bunit')
        info = {}
        info['file'] = filename
        info['object'] = hdr['OBJECT'].strip()
        info['beam'] = Beam(fwhm=[hdr['BMAJ'], hdr['BMIN']]*u.deg,
                            pa=hdr['BPA']*u.deg)
        info['frequency'] = hdr['RESTFRQ']*u.Hz
        info['date_obs'] = time.Time(hdr['DATE-OBS'])
        info['skycoord'] = SkyCoord(ra=hdr['OBSRA']*u.deg,
                                    dec=hdr['OBSDEC']*u.deg,
                                    frame=hdr['RADESYS'].lower(),
                                    **skycoord_kwargs)
        info['wcs'] = WCS(hdr, **wcs_kwargs)
        info['xscl'] = abs(hdr['cdelt1']*u.deg)
        info['yscl'] = abs(hdr['cdelt2']*u.deg)
        return cls(data, meta=info, header=hdr)

    def calc_geometry(self, metakernel=None):
        """Calculate the observing and illumination geometry

        Parameters
        ----------
        metakernel : str
            NAIF SPICE meta kernel to be loaded

        The target is from `.meta['object']`.  The calculation of geometry
        uses SPICE kernels specified by `metakernel` or preloaded before
        function call.
        """
        if metakernel is not None:
            spice.furnsh(metakernel)
            self.meta['metakernel'] = metakernel
            if self.header is not None:
                self.header['metakern'] = metakernel
        if 'utc_mid' in self.meta.keys():
            ut = self.meta['utc_mid'].isot
        else:
            ut = self.meta['date_obs'].isot
        target = self.meta['object']
        self.meta['geom'] = utils.subcoord(ut, target)
        if self.header is not None:
            for k in self.meta['geom'].keys():
                self.header[k] = self.meta['geom'][0][k]
        if metakernel is not None:
            spice.unload(metakernel)

    def disp(self, ds9=None, **kwargs):
        """Display image in DS9

        Parameters
        ----------
        ds9 : `saoimage.DS9`, optional
            DS9 window to display the image.  If `None`, then a new DS9 window
            will be opened.
        kwargs : dict
            Keyword arguments for `saoimage.Region`.  This can be used to set
            the properties of beam ellipse, such as color and width.
        """
        if ds9 is None:
            ds9=saoimage.getds9()
        ds9.imdisp(self)
        # beam parameter
        bmaj = (self.meta['beam'].fwhm_major/self.meta['xscl']).to('').value
        bmin = (self.meta['beam'].fwhm_minor/self.meta['xscl']).to('').value
        bpa = self.meta['beam'].pa.to('deg').value
        # display parameter
        sy, sx = self.shape
        w = int(ds9.get('width'))
        h = int(ds9.get('height'))
        z = float(ds9.get('zoom'))
        p = np.array([float(x) for x in ds9.get('pan').split()]) - 1
        orig = p - np.array([w/2, h/2])/z
        # generate beam marker
        bct = np.array([20, 20]) + bmaj*2*z
        bx, by = bct/z + orig
        bx = np.clip(bx, p[0]-w/2/z, p[0])
        by = np.clip(by, p[1]-h/2/z, p[1])
        beam = saoimage.EllipseRegion(bx, by, bmaj, bmin, bpa, **kwargs)
        beam.show(ds9)

    def get_meta(self, keys=None, wcs=False):
        """Return meta data contained in the object

        Parameters
        ----------
        keys : list of str, optional
            List of meta data keys to be returned.  If `None`, all keys
            except for `'wcs'` are returned.
        wcs : bool, optional
            If `True`, return all WCS keys.

        Returns
        -------
        dict : `utils.MetaData`
            Meta data items contained in the object
        """
        meta = self.meta.copy()
        w = meta.pop('wcs', None)
        if keys is None:
            keys = list(meta.keys())
        if 'wcs' in keys:
            keys.remove('wcs')
        if 'skycoord' in keys:
            coord = meta.pop('skycoord', None)
            keys.remove('skycoord')
        else:
            coord = None
        if 'beam' in keys:
            beam = meta.pop('beam', None)
            keys.remove('beam')
        else:
            beam = None
        if 'geom' in keys:
            geom = meta['geom']
            keys.remove('geom')
        else:
            geom = None
        from .utils import MetaData
        out = MetaData()
        for k in keys:
            setattr(out, k, meta[k])
        if coord is not None:
            setattr(out, 'ra', coord.ra)
            setattr(out, 'dec', coord.dec)
        if beam is not None:
            setattr(out, 'bmaj', beam.fwhm_major)
            setattr(out, 'bmin', beam.fwhm_minor)
            setattr(out, 'bpa', beam.pa)
            setattr(out, 'barea', beam.area)
        if geom is not None:
            for k in geom.keys():
                d = geom[k].data[0]
                u = geom[k].unit
                if u is not None:
                    d = d*u
                setattr(out, k, d)
        if wcs:
            for k, v in w.to_header().items():
                setattr(out, k, v)
        return out

    def to_fits(self, filename=None, overwrite=False):
        """Write image to FITS file"""
        if filename is None:
            filename = self.info['file']
        hdu = fits.PrimaryHDU(np.asarray(self), self.header)
        hdu.header.insert('object', ('bunit', str(self.unit),
            'Brightness (pixel) unit'), after=True)
        hdu.header.insert('bunit', ('bscale', 1.), after=True)
        hdu.header.insert('bunit', ('bzero', 0.), after=True)
        hdu.writeto(filename, overwrite=overwrite)


class ALMACeresImage(ALMAImage):
    """Subclass of `ALMAImages` for specific functionalities of Ceres images
    """

    ceres = Ceres()

    def centroid(self, box=None, method=1, **kwargs):
        """Find the centroid of Ceres

        Parameters
        ----------
        box : number, optional
            Box size to measure centroid.  Default is half of the shorter
            dimension of the image.
        method : [0, 1, 2], optional
            Specify the method to measure centroid.  See `utils.centroid`.
        kwargs : dict, optional
            Other keywords accepted by `utils.centroid`.

        Method sets `.meta['center'] = cy, cx`.
        """
        if box is None:
            if 'geom' in self.meta.keys():
                angular_dia = self.ceres.shape.a / self.meta['geom']['Range']
                angular_x = (angular_dia / self.meta['xscl']).to('',
                                equivalencies=u.dimensionless_angles()).value
                angular_y = (angular_dia / self.meta['yscl']).to('',
                                equivalencies=u.dimensionless_angles()).value
                box = max([angular_x, angular_y]) * 4
            else:
                box = min(self.shape)/2.1
        if (method == 2) and ('threshold' not in kwargs.keys()):
            _, std = utils.resmean(self, std=True)
            threshold = std*5
            kwargs['threshold'] = threshold
        self.meta['center'] = utils.centroid(self, method=method, box=box,
                                             **kwargs)

    def set_obstime(self, utc_start, utc_stop):
        """Set utc_start and utc_stop times.

        The start and stop UTC of each images are usually provided separately
        based on the processing of each image.
        """
        self.meta['utc_start'] = time.Time(utc_start)
        self.meta['utc_stop'] = time.Time(utc_stop)
        self.meta['utc_mid'] = self.meta['utc_start'] + \
                (self.meta['utc_stop'] - self.meta['utc_start']) / 2
        if self.header is not None:
            self.header['UTCSTART'] = self.meta['utc_start'].isot
            self.header['UTCSTOP'] = self.meta['utc_stop'].isot
            self.header['UTCMID'] = self.meta['utc_mid'].isot

    def project(self, return_all=False):
        """Project the image into lon-lat projection

        Parameters
        ----------
        return_all : bool, optional
            If `True`, returns all relevant projections

        Returns
        -------
        b, or tuple (b, lst, emi, lat, lon)
        All returned variables are `LonLatProjection`
        b : Projected image
        lst : Local solar time
        emi : Emission angle
        lat : Latitude
        lon : Longitude
        """
        if 'center' not in self.meta.keys():
            raise ValueError('Centroid not found.  Please run '
                             '`self.centroid` first.')

        lat, lon = utils.makenxy(-90,90,91,0,358,180)
        vpt = vector.Vector(1., self.meta['geom']['SOLon'][0],
                            self.meta['geom']['SOLat'][0], type='geo',
                            deg=True)
        pxlscl = (self.meta['geom']['Range'] * self.meta['xscl']).to('km',
                 equivalencies=u.dimensionless_angles()).value
        center = self.meta['center']
        x, y = vector.lonlat2xy(lon, lat, self.ceres.shape.r.to('km').value,
                                vpt, pa=self.meta['geom']['PolePA'],
                                center=center, pxlscl=pxlscl)
        w = np.isfinite(x) & np.isfinite(y)
        b = u.Quantity(np.zeros_like(x), unit=self.unit)
        b[w] = self[np.round(y[w]).astype(int), np.round(x[w]).astype(int)]
        b = LonLatProjection.from_array(b)

        if return_all:
            # calculate local solar time
            lst = ((lon - self.meta['geom']['SSLon']) / 15 + 12) % 24 * u.hour
            lst = LonLatProjection.from_array(lst)

            # calculate emission angle
            latlon_vec = vector.Vector(np.ones_like(lon), lon, lat, type='geo',
                                       deg=True)
            subsolar_vec = vector.Vector(1, self.meta['geom']['SOLon'][0],
                                         self.meta['geom']['SOLat'][0],
                                         type='geo', deg=True)
            emi = latlon_vec.vsep(subsolar_vec) * u.deg
            emi = LonLatProjection.from_array(emi)

            # lat and lon array
            lat = LonLatProjection.from_array(lat*u.deg)
            lon = LonLatProjection.from_array(lon*u.deg)

            return b, lst, emi, lat, lon
        else:
            return b

    def get_meta(self, keys=None, wcs=False):
        """Return meta data contained in the object.

        See `ALMAImage.get_meta`.  This function adds `.meta['center']` key
        to the output list.
        """
        meta = self.meta.copy()
        if keys is None:
            keys = list(self.meta.keys())
        if 'center' in keys:
            center = meta['center']
            keys.remove('center')
        else:
            center = None
        out = super().get_meta(keys=keys, wcs=wcs)
        if center is not None:
            setattr(out, 'cx', center[1])
            setattr(out, 'cy', center[0])
        return out


class LonLatProjection(u.Quantity):
    """Longitude-Latitude projection class
    """
    def __new__(cls, input_array, meta=None, **kwargs):
        obj = u.Quantity(input_array, **kwargs).view(cls)
        obj.meta = meta
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None: return
        self.meta = getattr(obj, 'meta', None)

    @classmethod
    def from_array(cls, input_array, lonlim=[0, 360], latlim=[-90, 90], **kwargs):
        """Initialize class from an array or a `astropy.units.Quantity`"""
        input_array = np.asanyarray(input_array)
        info = {}
        info['lonlim'] = lonlim
        info['latlim'] = latlim
        if len(input_array.shape) == 2:
            info['nband'] = 1
        else:
            info['nband'] = input_array.shape[0]
        return cls(input_array, meta=info, **kwargs)

    def to_fits(self, filename=None, extname=None, append=False,
                overwrite=False):
        """Save projection to FITS file

        Parameters
        ----------
        filename : str, optional
            File name to save the FITS
        append : bool, optional
            Append to existing FITS file
        overwrite : bool, optional
            Overwrite existing FITS file

        Return
        ------
        `astropy.io.fits.ImageHDU`
        """
        hdu = fits.ImageHDU(np.asarray(self))
        if (self.unit is not None) and (str(self.unit) != ''):
            hdu.header['bunit'] = str(self.unit)
        hdu.header['lonmin'] = self.meta['lonlim'][0]
        hdu.header['lonmax'] = self.meta['lonlim'][1]
        hdu.header['latmin'] = self.meta['latlim'][0]
        hdu.header['latmax'] = self.meta['latlim'][1]
        if filename is not None:
            from os.path import isfile
            if append:
                overwrite = True
                if extname is not None:
                    hdu.header['EXTNAME'] = extname
                if isfile(filename):
                    hdulist = fits.open(filename)
                else:
                    hdulist = fits.HDUList()
            else:
                hdulist = fits.HDUList()
            hdulist.append(hdu)
            hdulist.writeto(filename, overwrite=overwrite)
        return hdu

    def concatenate(self, proj, ignore_unit=False, equivalencies=None, axis=0):
        """Concatenate another `LonLatProjection` or array-like

        proj : `LonLatProjection`, array-like
            Array to be concatenated
        ignore_unit : bool, optional
            If `True`, ignore unit of `proj` if incompatible with `self`
        equivalencies : list
            List of equivalencies to suppoert unit conversion of `proj` before
            concatenate.
        axis : int, optional
            Axis to be concatenated.  See `numpy.concatenate`.

        Return
        ------
        Concatenated array or object

        Note
        ----
        Concatenation follows all the rules of `numpy.concatenate`.  This
        method checks and tries to convert unit of `proj` before concatenation.
        If unit is incompatible, unless `ignore_unit is True`, a `ValueError`
        will be raised.  `self.meta` will be copied to output.
        """
        if isinstance(proj, LonLatProjection) and hasattr(proj, 'meta'):
            if (('lonlim' in proj.meta.keys()) and \
                              (self.meta['lonlim'] != proj.meta['lonlim'])) \
                or (('latlim' in proj.meta.keys()) and \
                              (self.meta['latlim'] != proj.meta['latlim'])):
                    raise ValueError('Cannot concatenate with a projection'
                                     ' with different lon-lat limit.')
        if (hasattr(proj, 'unit') and (self.unit != proj.unit)):
            try:
                proj = proj.to(self.unit, equivalencies=equivalencies)
            except u.UnitConversionError:
                if not ignore_unit:
                    raise ValueError('Incompatible unit for concatenation.')
        p1 = np.asarray(self)
        p2 = np.asarray(proj)
        out = LonLatProjection.from_array(np.concatenate([p1,p2], axis=axis),
                                          unit=self.unit)
        out.meta = self.meta.copy()
        return out


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
    """Interferometer beam class

    Attributes
    ----------
    fwhm : Full-width half-max
    fwhm_major : Full-width half-max full major axis
    fwhm_minor : Full-width half-max full minor axis
    sigma : Beam sigma
    sigma_major : Beam sigma in the major axis
    sigma_minor : Beam sigma in the minor axis
    pa : Position angle (north to east) of the major axis
    area : Beam area
    """
    @u.quantity_input(pa=u.deg)
    def __init__(self, fwhm=None, sigma=None, pa=None):
        """Define beam size

        fwhm : `astropy.units.Quantity` or number, or [major, minor]
            The full-width half-max.
        sigma : `astropy.units.Quantity' or number, or [major, minor]
            The beam sigma.
        pa : `astropy.units.Quantity` with a unit of angle
            Position angle of the major axis

        Either `fwhm` or `sigma`, but not both, needs to be specified.
        Otherwise an `ValueError` will be raised.  If a single value
        is specified, then the beam is assumed to be circular (major and
        minor axes are the same).

        No restrictions on the unit of beam size are imposed, and no check on
        the unit is done.  Caller is responsible for ensuring the physical
        meaningfulness of the unit.
        """
        if (fwhm is None) and (sigma is None):
            raise ValueError('Beam width is not specified.')
        if (fwhm is not None) and (sigma is not None):
            raise ValueError('Only one of `fwhm` or `sigma` can be specified.')
        if fwhm is not None:
            try:
                n = len(fwhm)
                if n == 1:
                    self._fwhm = [fwhm[0], fwhm[0]]
                else:
                    self._fwhm = fwhm
            except TypeError:
                self._fwhm = [fwhm, fwhm]
        else:
            try:
                n = len(sigma)
                if n == 1:
                    self._sigma = [sigma[0], sigma[0]]
                else:
                    self._sigma = sigma
            except:
                self._sigma = [sigma, sigma]
        if pa is not None:
            self.pa = pa

    @property
    def fwhm(self):
        if hasattr(self, '_fwhm'):
            return self._fwhm
        else:
            return [self._sigma[0] * 2.3548200450309493, \
                    self._sigma[1] * 2.3548200450309493]

    @property
    def sigma(self):
        if hasattr(self, '_sigma'):
            return self._sigma
        else:
            return [self._fwhm[0] * 0.42466090014400953, \
                    self._fwhm[1] * 0.42466090014400953]

    @property
    def fwhm_major(self):
        return self.fwhm[0]

    @property
    def fwhm_minor(self):
        return self.fwhm[1]

    @property
    def sigma_major(self):
        return self.sigma[0]

    @property
    def sigma_minor(self):
        return self.sigma[1]

    @property
    def area(self):
        """Beam area"""
        return 2 * np.pi * self.sigma_major * self.sigma_minor


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
        bm = Beam([hdr['BMAJ']*u.deg,hdr['BMIN']*u.deg]).area.to('arcsec2').value
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


def project(metadata, rc=ceres.shape.r.value, saveto=None):
    """Project images to lat-lon projection
    """
    # load image
    fname = path.join(metadata.path, metadata.name)
    im = utils.readfits(fname, verbose=False)
    im = np.squeeze(im)
    im /= Beam([metadata.bmaj*u.mas,metadata.bmin*u.mas]).area.to('arcsec2').value  # in Jy/arcsec**2

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
    """Implementation of Snell's law of reflection.

    The initialization takes two refractive indexes, and the default for
    the refractive index of the second medium is 1.

    >>> s = Snell(1.5)
    >>> print(s.n1)
    1.5
    >>> print(s.n2)
    1.0

    The methods `.angle1` and `.angle2` calculate the angle in medium 1 and
    2, respectively, from the angle in the other medium.
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

    @property
    def brewster_angle(self):
        """Brewster's angle

        Calculated for light transmiting from medium 1 (n1) to medium 2 (n2)
        """
        return np.rad2deg(np.arctan(self.n2/self.n1))

    def reflectance_coefficient(self, angle1=None, angle2=None, pol=None):
        """Calculate reflectance coefficient

        Parameter
        ---------
        angle1 : number, `astropy.units.Quantity`
            Angle in the 1st medium.  Only one of `angle1` or `angle2` should
            be passed.  If both passed, then an error will be thrown.
        angle2 : number, `astropy.units.Quantity`
            Angle in the 2nd medium.  Only one of `angle1` or `angle2` should
            be passed.  If both passed, then an error will be thrown.
        pol : in ['s', 'normal', 'perpendicular', 'p', 'in plane', 'parallel']
            The polarization state for calculation.
            ['s', 'normal', 'perpendicular']: normal to plane of incidence
            ['p', 'in plane', 'parallel']: in the plan of incidence
            Default will calculate the average of both polarization states

        Return
        ------
        Reflection coefficient
        """
        if angle1 is not None and angle2 is not None:
            raise ValueError('ony one angle should be passed')
        if angle1 is not None:
            angle2 = self.angle2(angle1)
        if angle2 is not None:
            angle1 = self.angle1(angle2)
        if pol in ['s', 'normal', 'perpendicular', None]:
            a = self.n1 * np.cos(np.deg2rad(angle1))
            b = self.n2 * np.cos(np.deg2rad(angle2))
            Rs = ((a - b)/(a + b))**2
        if pol in ['p', 'in plane', 'parallel', None]:
            a = self.n1 * np.cos(np.deg2rad(angle2))
            b = self.n2 * np.cos(np.deg2rad(angle1))
            Rp = ((a - b)/(a + b))**2
        try:
            return (Rs + Rp)/2
        except NameError:
            try:
                return Rs
            except NameError:
                return Rp


class Layer(object):
    """Layer class for calculating propagation of subsurface thermal emission

    Based on the models in Keihm & Langseth (1975), Icarus 24, 211-230
    """

    def __init__(self, n, loss_tangent, depth=np.inf, profile=None):
        """
        Parameters
        ----------
        n : number
            Real part of refractive index
        loss_tangent : number
            Loss tengent
        depth : optional, number
            Depth of the layer in the same unit as `z` (see below for the
            description of `profile`)
        profile : optional, callable object
            `profile(z)` returns the physical quantity of the layer at depth
            `z`.  Most commonly it is a temperature profile, or a thermal emission profile.
        """
        self.n = n
        self.depth = depth
        self.loss_tangent = loss_tangent
        self.profile = profile

    def absorption_length(self, *args, **kwargs):
        """
        Calculates absorption length

        See `.absorption_coefficient`
        """
        return 1./self.absorption_coefficient(*args, **kwargs)

    def absorption_coefficient(self, wave_freq):
        """
        Calculates absorption coefficient

        wave_freq : number, `astropy.units.Quantity` or array_like
            Wavelength or frequency of observation
        """
        if isinstance(wave_freq, u.Quantity):
            wavelength = wave_freq.to(u.m, equivalencies=u.spectral())
        else:
            wavelength = wave_freq
        c = np.sqrt(1+self.loss_tangent*self.loss_tangent)
        return (4*np.pi*self.n)/wavelength*np.sqrt((c-1)/(c+1))


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
            snell = Snell(l.n, n0)
            inc = snell.angle1(emi_ang)
            ref_coef = snell.reflectance_coefficient(angle2=emi_ang)
            coef = l.absorption_coefficient(wavelength)
            cos_i = np.cos(np.deg2rad(inc))
            if debug:
                print(f'cos(i) = {cos_i}, coef = {coef}')
            intfunc = lambda z: l.profile(z) * np.exp(-coef*z/cos_i-L)
            dd = -2.3026*np.log10(epsrel)/coef
            if l.depth > dd:
                zz = np.linspace(0, dd, 1000)
            else:
                zz = np.linspace(0, l.depth, 1000)
            if debug:
                prof['t'].append(l.profile(zz))
                prof['intprofile'].append(intfunc(zz))
                prof['zzz'].append(zz+D)
                prof['L0'].append(L)
                D += l.depth
            from scipy.integrate import quad
            # integral = quad(intfunc, 0, l.depth, epsrel=epsrel)[0]
            integral = (intfunc(zz)[:-1]*(zz[1:]-zz[:-1])).sum()
            m += (1-ref_coef)*coef*integral/cos_i
            # prepare for the next layer
            L += l.depth/cos_i*coef
            emi_ang = inc
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

            nt : int
                Number of time steps in one thermal cycle.
            dz : float
                Dimensionless depth step size in unit of thermal skin depth.
            z1 : int
                Dimensionless depth boundary in unit of thermal skin depth.
            init : 2d array_like of float
                Initial 2-D temperature distribution.  Same shape as the
                returned temperature model.  Default is 0.6*max(cos(inc)) for
                all depth and times.
            maxiter : int
                Maximum number of iterations (thermal cycles).
            tol : float
                Tolerance of convergence.
            inc : 1d array_like float or `astropy.units.Quantity`
                Sequence of solar incidence angle (`cos=False`) or the cosines
                thereof (`cos=True`) in one thermal cycle.  It must have the
                same shape as `nt`.  User has to ensure that incidence angle is
                physically meaningful, i.e., <90 deg or `cos(inc)>0`.
            cos : bool
                Specify whether `inc` is incidence angles or the cosines of.
            verbose : bool
                Enable verbose mode for the program to print out information
                and progress after each iteration.
            benchmark : bool
                If `True`, program will run with keywords plot and verbose
                disabled, and print out the CPU time in seconds used for
                calculation.  This keyword can be used to benchmark the speed
                of calculation.

        Returns
        -------
        None.  Upon return, this method will populate two class attribute,
        `.temperature_model` and `.model_param`

        `.temperature_model` saves the model temperature with respect to time
        and depth.

        `.model_param` is a `dict` that stores the model parameters.  The items
        are:
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

    def save(self, file):
        pass


class SphereTPM():
    """TPM model on a sphere

    Class object is initialized by a thermal physical model, the sub-solar
    latitude, the latitude grid size, and calculates the temperautre model.

    The temperature model is saved in property `.temperature_model`, and the
    model parameters are saved in property `.model_param`.

    `.temperature_model` : 3d `astropy.units.Quantity` of shape (nt, nz, nlat)
        The temperature model in (time, depth, latitude) grid.
        `nt` : number of time steps
        `nz` : number of depth steps
        `nlat` : number of latitudes

    `.model_param` : dict
        See `Thermal.model_param`
    """

    @u.quantity_input
    def __init__(self, tpm=None, sunlat: u.deg=0*u.deg, dlat: u.deg=5*u.deg):
        """
        Initialize class object

        Parameters
        ----------
        tpm : `Thermal`
            The thermophysical model class object used to calculate temperature
            model
        sunlat : `astropy.units.Quantity`
            Sub-solar latitude
        dlat : `astropy.units.Quantity`
            Latitude grid spacing
        """
        self.tpm = tpm
        self.sunlat = sunlat
        self.dlat = dlat

    def thermal_model(self, nt=360, dz=0.5, z1=50, init=None, maxiter=10000,
            tol=1e-4, verbose=False):
        """Calculate temperature model

        Parameters
        ----------
        See `Thermal.thermal_model()` for parameters

        Returns
        -------
        None.  Method updates property `.temperature_model`
        """

        if self.tpm is None:
            raise ValueError('no thermal model is defined')

        if self.tpm.Theta is None:
            raise ValueError('thermal parameter is unknown')

        self.model_param = {}
        # number of steps in depth
        nz = int(np.ceil(z1/dz)+1)
        # number of latitudinal zones
        nlat = int(180/self.dlat.to('deg').value+1)
        # latitudes of all zones
        lats =  np.linspace(-90, 90, nlat)
        self.model_param['latitude'] = lats * u.deg

        # initial condition
        if init is not None:
            tt0 = np.asarray(init)
            if tt0.shape != (nt, nz, nlat):
                warnings.warn('invalid initial condition is ignored')
                tt0 = np.ones((nt, nz, nlat)) * 0.4
        else:
            tt0 = np.ones((nt, nz, nlat)) * 0.4
        tt = np.zeros_like(tt0) * u.K

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
            norm = vector.Vector(np.ones_like(lons), lons, np.repeat(lt, nt),
                    type='geo', deg=True)
            inc = np.clip(np.cos(sunvec.vsep(norm)), 0, None)

            # calculate temperature only when the sun raises above horizon
            # some time in a day
            if inc.max() > 1e-5:
                self.tpm.thermal_model(nt=nt, dz=dz ,z1=z1 ,inc=inc, cos=True,
                    verbose=verbose ,tol=tol, init=np.squeeze(tt0[...,i]),
                    maxiter=maxiter)
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
        if getattr(self, 'temperature_model', None) is None:
            warnings.warn('model not generated yet, no data to save')
            return
        hdu = fits.PrimaryHDU(self.temperature_model.to('K').value)
        if self.tpm.skin_depth is not None:
            hdu.header['sunlat'] = (self.sunlat.to('deg').value,
                'sub-solar latitude (deg)')
            hdu.header['skindep'] = (self.tpm.skin_depth.to('m').value,
                'thermal skin depth (m)')
        if self.tpm.Period is not None:
            hdu.header['period'] = (self.tpm.Period.to('s').value,
                'thermal cycle period (sec)')
        hdu.writeto(file, overwrite=overwrite)
        utils.writefits(file, self.model_param['t'].to('s').value, name='Time',
            append=True)
        utils.writefits(file, self.model_param['lst'].to('hour').value,
            name='LST', append=True)
        utils.writefits(file, self.model_param['z'].to('m').value,
            name='Depth', append=True)
        utils.writefits(file, self.model_param['latitude'].to('deg').value,
            name='Latitude', append=True)
        utils.writefits(file, self.model_param['insolation'].to('W/m2').value,
            name='insol', append=True)
        utils.writefits(file, self.model_param['niter'], name='niter',
            append=True)

    def load(self, file):
        """Load temperature model and model parameters from input file
        """
        f = fits.open(file)
        self.sunlat = f[0].header['sunlat'] * u.deg
        self.temperature_model = f[0].data.copy() * u.K
        self.model_param = {}
        self.model_param['t'] = f['time'].data.copy() * u.s
        self.model_param['z'] = f['depth'].data.copy() * u.m
        self.model_param['lst'] = f['lst'].data.copy() * u.hour
        self.model_param['latitude'] = f['latitude'].data.copy() * u.deg
        self.model_param['insolation'] = f['insol'].data.copy() \
            * u.Unit('W/m2')
        self.model_param['niter'] = f['niter'].data.copy()

    @classmethod
    def from_file(cls, file):
        """Initialize class object from input file
        """
        out = cls()
        out.load(file)
        return out


class TriaxialThermalImage():
    """Simulate the thermal image of a triaxial shape

    """

    @u.quantity_input
    def __init__(self, shape=None, tpm=None, pixel_size: [u.km, None]=None,
                image_size=512):
        """Initialize class object

        Parameters
        ----------
        shape : `TriaxialShape`
            The shape of body to be simulated.  Default is a sphere of size
            100 km
        tpm : `SphereTPM`, str
            If `SphereTPM` : the object that contains the surface temperature
            model to be used to calculate thermal image.
            If `str` : the file name of surface temperature model generated and
            saved by `SphereTPM` class object
            This property has to be set before any simulation can be performed
        """
        if shape is None:
            self.shape = TriaxialShape(100*u.km, 100*u.km, 100*u.km)
        else:
            self.shape = shape
        if isinstance(tpm, SphereTPM):
            self.tpm = tpm
        elif isinstance(tpm, str):
            self.tpm = SphereTPM.from_file(tpm)
        else:
            self.tpm = None
        self.image_size = image_size
        if pixel_size is None:
            self.pixel_size = 512 / (self.shape.r.max() * 2.4)
        else:
            self.pixel_size = pixel_size

    @u.quantity_input
    def temperature(self, obs_lat: u.deg, obs_lst: [u.hour, u.deg]):
        """Calculate temperature image cube

        Parameters
        ----------
        obs_lat : `astropy.units.Quantity`
            Sub-observer latitude, must be between [-90, 90] deg.
        obs_lst : `astropy.units.Quantity`
            Sub-observer local solar time, must be between [0, 24] hours
            or [0, 360] deg.
        pixel_size : `astropy.units.Quantity`
            Pixel size (length scale) at the object to be simulated.  Default
            is such that the size of the simulated image is 1.2x the largest
            dimension of the shape
        image_size : number
            The size of simulated image
        """
        pass


class AverageBrightnessTemperature(u.SpecificTypeQuantity):
    """Disk averaged temperature"""

    _equivalent_unit = u.K

    @u.quantity_input(wavelength=[u.m, None], equivalencies=u.spectral())
    def __new__(cls, value, wavelength=None, surface=None, **kwargs):
        """Initialize DiskAverageTemperature

        Parameters
        ----------
        wavelength : `astropy.units.Quantity
            Wavelength or frequency or the equivalent quantity corresponding
            to the brightness temperature
        surface : `Surface`
            The `Surface` object that does the subsurface calculation.
        **kwargs : other keywords to initialize `astropy.units.Quantity` object
        """
        T = super().__new__(cls, value, **kwargs)
        if wavelength is not None:
            T._wavelength = wavelength.to(u.m, equivalencies=u.spectral())
        else:
            T._wavelength = None
        T.surface = surface
        return T

    @property
    def wavelength(self):
        """Wavelength"""
        return self._wavelength

    @property
    def frequency(self):
        """Frequency"""
        return self._wavelength.to(u.Hz, equivalencies=u.spectral())

    @classmethod
    @u.quantity_input(wavelength=u.m, equivalencies=u.spectral())
    def from_model(cls, files, surface, wavelength, savemap=None,
            overwrite=True, benchmark=False,):
        """Calculate disk-averaged temperature from simulated images

        Parameters
        ----------
        files : `str`
            The names of FITS files that contain the temperature image or cube.
            If 2d image, then the average will simply be taken as the average
            over the disk.
            If 3d cube, then the first dimension is taken as the depth.  In
            this case, the first extension of the FITS file should store the
            depth array, and the second dimension the emission angle array.
        surface : `Surface`
            The `Surface` object that does the subsurface calculation.
        wavelength : `astropy.units.Quantity
            Wavelength or frequency or the equivalent quantity corresponding
            to the brightness temperature

        Returns
        -------
        An `AverageBrightnessTemperature` object with each element
        calculated from the input file.
        """
        # initialize subsurface layer
        if savemap is not None:
            savemap = savemap.split('.')

        wavelength = wavelength.to('m', u.spectral())
        if isinstance(files, str):
            files = [files]
        t = np.empty(len(files))
        for i,f in enumerate(files):
            print(i, path.basename(f))
            im = fits.open(f)
            if len(im[0].data.shape) == 2:
                inside = im[0].data > 0
                t[i] = im[0].data[inside].mean()
            elif len(im[0].data.shape) == 3:
                t_eff = np.zeros_like(im[0].data[0])
                inside = im[0].data[0] > 0
                if len(np.where(inside)[0]) > 71400:
                    warnings.warn('computational time could be > 30 s')
                if benchmark:
                    print(len(np.where(inside)[0]))
                    j=0
                    import time
                    t0 = time.time()
                for i1, i2 in np.array(np.where(inside)).T:
                    if benchmark:
                        if j % 1000 == 0:
                            t1 = time.time()
                            print(j, t1-t0)
                            t0 = t1
                    from scipy.interpolate import interp1d
                    surface.layers[0].profile = interp1d(im[1].data,
                        im[0].data[:,i1,i2], kind='cubic', bounds_error=False,
                        fill_value=(None, im[0].data[-1,i1,i2]))
                    t_eff[i1, i2] = surface.emission(im[2].data[i1, i2],
                        wavelength.value)
                    if benchmark:
                        j+=1
                t[i] = t_eff[inside].mean()

                if savemap is not None:
                    filename = savemap.copy()
                    filename[-2] = filename[-2]+f'_{i:03d}'
                    filename = '.'.join(filename)
                    utils.writefits(filename, t_eff, overwrite=overwrite)

        return cls(t*u.K, wavelength=wavelength, surface=surface)
