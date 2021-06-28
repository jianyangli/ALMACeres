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
import spiceypy as spice
from jylipy.image import ImageSet
from jylipy import shift
from xarray import DataArray


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
        if ('geometry' in hdr) and hdr['geometry']:
            keys = ['time', 'rh', 'range', 'phase', 'ra', 'dec', 'solat',
                    'solon', 'sslat', 'sslon', 'polepa', 'poleinc', 'sunpa',
                    'suninc']
            vals = [[hdr[k]] for k in keys]
            info['geom'] = table.Table(vals, names=keys)
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
        geom = utils.subcoord(ut, target)
        for k in geom.keys():
            geom.rename_column(k, k.lower())
        self.meta['geom'] = geom
        if self.header is not None:
            self.header['geometry'] = True
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

    @classmethod
    def from_fits(cls, *args, **kwargs):
        obj = ALMAImage.from_fits(*args, **kwargs)
        #print(args)
        if isinstance(args[0], str):
            hdr = fits.open(args[0])[0].header
        elif isinstance(args[0], fits.HDUList):
            hdr = args[0][0].header
        if ('CTR_X' in hdr) and ('CTR_Y' in hdr):
            obj.meta['center'] = hdr['CTR_Y'], hdr['CTR_X']
        if 'UTCSTART' in hdr:
            obj.meta['utc_start'] = time.Time(hdr['UTCSTART'])
        if 'UTCSTOP' in hdr:
            obj.meta['utc_stop'] = time.Time(hdr['UTCSTOP'])
        if 'UTCMID' in hdr:
            obj.meta['utc_mid'] = time.Time(hdr['UTCMID'])
        return obj.view(ALMACeresImage)

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
        self.set_center(utils.centroid(self, method=method, box=box, **kwargs))

    def set_center(self, center):
        self.meta['center'] = center
        self.header['CTR_X'] = self.meta['center'][1]
        self.header['CTR_Y'] = self.meta['center'][0]

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
        vpt = vector.Vector(1., self.meta['geom']['solon'][0],
                            self.meta['geom']['solat'][0], type='geo',
                            deg=True)
        pxlscl = (self.meta['geom']['range'] * self.meta['xscl']).to('km',
                 equivalencies=u.dimensionless_angles()).value
        center = self.meta['center']
        x, y = vector.lonlat2xy(lon, lat, self.ceres.shape.r.to('km').value,
                                vpt, pa=self.meta['geom']['polepa'][0],
                                center=center, pxlscl=pxlscl)
        w = np.isfinite(x) & np.isfinite(y)
        b = u.Quantity(np.zeros_like(x), unit=self.unit)
        b[w] = self[np.round(y[w]).astype(int), np.round(x[w]).astype(int)]
        b = LonLatProjection.from_array(b)

        if return_all:
            # calculate local solar time
            lst = ((lon - self.meta['geom']['sslon']) / 15 + 12) % 24 * u.hour
            lst = LonLatProjection.from_array(lst)

            # calculate emission angle
            latlon_vec = vector.Vector(np.ones_like(lon), lon, lat, type='geo',
                                       deg=True)
            subsolar_vec = vector.Vector(1, self.meta['geom']['solon'][0],
                                         self.meta['geom']['solat'][0],
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


class ProcessingALMACeresImages():
    def __init__(self, meta):
        """
        Parameters
        ----------
        meta : `ALMACeres.utils.MetaData`
            The meta data object loaded from the JSON file corresponding to each dataset
            JSON data files are saved in ALMACeres/data/ directory

        Attributes for data processing:
            flux_scaling : number or iterable of numbers
                Flux scaling factors
            projected_dir : str
                The directory to save projected images
            preprocess_dir : str
                The directory to save pro-processed images
        These attributes need to be preset before the relevant processing is performed:
        """
        self.meta = meta
        self.flux_scaling = 1.
        self.projected_dir = None
        self.preprocess_dir = None

    def preprocess(self, infile, outfile=None, scaling=1, utc_start=None, utc_stop=None):
        """
        Preprocess image generated by pipeline

        fitsname : str
            FITS file name
        scaling : number
            Flux scaling factor

        Processing performed:
        1. Apply flux scaling factor
        2. Convert to unit of Jy/arcsec2
        3. Calculate observing geometry

        The calculation of geometry requires SPICE kernels to be pre-loaded.
        """
        hdus = fits.open(infile)
        hdus[0].header['object'] = 'Ceres'  # some images have wrong 'object' key
        im = ALMACeresImage.from_fits(hdus)  # squeeze out the frequency and polarization dimension
        im.meta['file'] = path.basename(infile)  # fix file name
        im = im * scaling  # apply flux scaling from Arielle for new flux cal
        im = im.to('Jy/arcsec2', equivalencies=u.beam_angular_area(im.meta['beam'].area))  # convert to Jy/arcsec2
        if (utc_start is not None) and (utc_stop is not None):
            im.set_obstime(utc_start, utc_stop)  # set start and stop times
        im.calc_geometry()  # calculate observing geometry
        if outfile is not None:
            im.to_fits(outfile)
        else:
            return im

    def __call__(self, centers=None, overwrite=False):
        """Data processing pipeline

        The tasks completed here include:
            1. Pre-processing, including
                1.1 Apply flux scaling factor
                1.2 Convert to unit of Jy/arcsec2
                1.3 Calculate observing geometry
            2. Find or set disk-center
            3. Save processed image file to `.preprocess_dir/` under original file name
            4. Generate projected images.  It will save all projected images into
               directory specified by `.projected_dir` under the original file name, and
               save all projected images concatenated in '`.projected_dir/all_projected.fits'
            5. Save all meta data in a `ALMACeres.utils.MetaDataList` as an attribute
               self.metadata
        """
        spice.furnsh(self.meta.metakernel)
        spice.furnsh(self.meta.pck_dawn)

        files = [path.join(self.meta.datadir, x) for x in self.meta.files]
        nfiles = len(files)
        utc_start = self.meta.utc_start
        utc_stop = self.meta.utc_stop
        if (not hasattr(self.flux_scaling, '__iter__')) or (len(self.flux_scaling) == 1):
            flux_scaling = np.repeat(self.flux_scaling, nfiles)
        else:
            flux_scaling = self.flux_scaling

        metadata = utils.MetaDataList(ALMACeresImageMetaData)
        imgs = []
        lsts = []
        emis = []
        for i, f in enumerate(files):
            filebase = path.basename(f)
            print('processing {}'.format(filebase))
            outfile = path.join(self.preprocess_dir, filebase)
            im = self.preprocess(f, scaling=flux_scaling[i], utc_start=utc_start[i], utc_stop=utc_stop[i])
            if centers is None:
                im.centroid(method=2)
            else:
                im.set_center(centers[i])
            im.to_fits(outfile, overwrite=overwrite)
            im = np.squeeze(im)
            # calculate and save projected maps
            b, lst, emi, lat, lon = im.project(return_all=True)
            projfile = path.join(self.projected_dir, filebase)
            b.astype('float32').to_fits(projfile, overwrite=True)
            lst.astype('float32').to_fits(projfile, extname='LST', append=True)
            emi.astype('float32').to_fits(projfile, extname='EMI', append=True)
            lat.astype('float32').to_fits(projfile, extname='LAT', append=True)
            lon.astype('float32').to_fits(projfile, extname='LON', append=True)
            imgs.append(b)
            lsts.append(lst)
            emis.append(emi)
            metadata.append(im.get_meta())

        # save meta data
        for m in metadata:
            m.xscl = m.xscl.to('mas')
            m.yscl = m.yscl.to('mas')
            m.bmaj = m.bmaj.to('mas')
            m.bmin = m.bmin.to('mas')
            m.barea = m.barea.to('arcsec2')
            m.freqency = m.frequency.to('GHz')
        self.metadata = metadata

        # save all projections into one file
        print('saving projections')
        outfile = path.join(self.projected_dir, 'all_projected.fits')
        x = imgs[0].copy()[np.newaxis, ...]
        for i in imgs[1:]:
            x = x.concatenate(i[np.newaxis, ...])
        x.to_fits(outfile, overwrite=overwrite)

        x = lsts[0].copy()[np.newaxis, ...]
        for i in lsts[1:]:
            x = x.concatenate(i[np.newaxis, ...])
        x.to_fits(outfile, extname='LST', append=True)

        x = emis[0].copy()[np.newaxis, ...]
        for i in emis[1:]:
            x = x.concatenate(i[np.newaxis, ...])
        x.to_fits(outfile, extname='EMI', append=True)

        lat.to_fits(outfile, extname='LAT', append=True)
        lon.to_fits(outfile, extname='LON', append=True)


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
            the angle(s) in the first media.  If not Quantity, then in
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
        Calculates the refractive angle in the first media from that in the
        second media.

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
        if (angle1 is not None) and (angle2 is not None):
            raise ValueError('ony one angle should be passed')
        if (angle1 is None) and (angle2 is None):
            raise ValueError('one angle has to be passed')
        if angle1 is not None:
            angle2 = self.angle2(angle1)
        if angle2 is not None:
            angle1 = self.angle1(angle2)
        if pol in ['s', 'normal', 'perpendicular', None]:
            a = self.n1 * np.cos(np.deg2rad(angle1))
            b = self.n2 * np.cos(np.deg2rad(angle2))
            Rs = ((a - b)/(a + b))**2
            if not np.isfinite(Rs):
                Rs = 1.0
        if pol in ['p', 'in plane', 'parallel', None]:
            a = self.n1 * np.cos(np.deg2rad(angle2))
            b = self.n2 * np.cos(np.deg2rad(angle1))
            Rp = ((a - b)/(a + b))**2
            if not np.isfinite(Rp):
                Rp = 1.0
        try:
            return (Rs + Rp)/2
        except NameError:
            try:
                return Rs
            except NameError:
                return Rp


class Layer(object):
    """Layer class for calculating propagation of subsurface thermal emission

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
        profile : optional, callable object or array-like of shape (2, N)
            If callable, then `profile(z)` returns the physical quantity of
            the layer at depth `z`.  Most commonly it is a temperature
            profile, or a thermal emission profile.
            If array-like, then profile[0] is depth and profile[1] is the
            physical quantity at corresponding depth.
        """
        self.n = n
        self.depth = depth
        self.loss_tangent = loss_tangent
        if profile is None:
            self.profile = None
        elif hasattr(profile, '__call__'):
            self.profile = profile
        elif np.shape(profile)[0] == 2:
            from scipy.interpolate import interp1d
            self.profile = interp1d(profile[0], profile[1], bounds_error=False,
                                    fill_value=(profile[1][0], profile[1][-1]))
        else:
            raise ValueError('Unrecogniazed type or format for `profile`')

    def absorption_length(self, *args, **kwargs):
        """
        Calculates absorption length

        See `.absorption_coefficient`
        """
        return 1./self.absorption_coefficient(*args, **kwargs)

    def absorption_coefficient(self, wave_freq):
        """
        Calculates absorption coefficient.
        Follow Hapke (2012) book, Eq. 2.84, 2.85, 2.95b

        wave_freq : number, `astropy.units.Quantity` or array_like
            Wavelength or frequency of observation.  Default unit is
            'm' if not specified.
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

    def __init__(self, layers, profile=None):
        """
        layers: SubsurfaceLayer class object or array_like of it, the
            subsurface layers.  If array-like, then the layers are ordered
            from the top surface downward.
        profile : optional, array-like of shape (2, N)
            `profile[0]` is depth and `profile[1]` is the physical quantity at
            corresponding depth.
            If this parameter is specified, it will override the `.profile`
            properties of input layers.  This is the mechanism to provide a
            continuous temperature profile for multi-layered model.
        """
        if not hasattr(layers, '__iter__'):
            self.layers = [layers]
        else:
            self.layers = layers
        self.depth = 0
        self._check_layer_depth()
        self.depth = np.sum([l.depth for l in self.layers])
        self.n_layers = len(self.layers)
        self.profile = profile
        # set `.profile` properties for all layers
        if profile is not None:
            if hasattr(profile, '__call__'):
                for l in self.layers:
                    l.profile = profile
            elif np.shape(profile)[0] == 2:
                from scipy.interpolate import interp1d
                prof_int = interp1d(profile[0], profile[1], bounds_error=False,
                                    fill_value=(profile[1][0], profile[1][-1]))
                if self.n_layers == 1:
                    self.layers[0].profile = prof_int
                else:
                    z0 = 0
                    for l in self.layers:
                        l.profile = interp1d(profile[0]-z0, profile[1],
                                    bounds_error=False,
                                    fill_value=(profile[1][0], profile[1][-1]))
                        z0 += l.depth
            else:
                raise ValueError('Unrecogniazed type or format for `profile`')

    def _check_layer_depth(self):
        for i,l in enumerate(self.layers[:-1]):
            if l.depth == np.inf:
                raise ValueError('only the deepest layer can have infinite '
                    'depth.  The depth of the {}th layer cannot be '
                    'infinity'.format(i))


    def _check_layer_profile(self):
        for i,l in enumerate(self.layers):
            if l.profile is None:
                raise ValueError('the {}th layer does not have a quantity '
                    'profile defined'.format(i))
            if not hasattr(l.profile, '__call__'):
                raise ValueError('the {}th layer does not have a valid '
                    'quantity profile defined'.format(i))


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

        L = 0    # total path length of light in the unit of absorption length
        n0 = 1.  # adjacent media outside of the layer to be calculated
        m = 0.   # integrated quantity
        trans_coef = 1.  # transmission coefficient = 1 - ref_coef
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
            intfunc = lambda z: l.profile(z) * np.exp(-coef*z/cos_i-L)
            dd = -2.3026*np.log10(epsrel)/coef
            #dd = l.profile.x.max()
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
            trans_coef *= (1-ref_coef)
            m += trans_coef*coef*integral/cos_i
            # prepare for the next layer
            L += l.depth/cos_i*coef
            emi_ang = inc
            n0 = l.n
            if debug:
                print(f'cos(i) = {cos_i}, coef = {coef}, ref_coef = {ref_coef}, integral = {integral}')

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


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
class PCAModelFitting(PCA):
    """PCA model fitting class
    """
    def __init__(self, model, ti, emis, icedep, loss,
                                lst=None, nonzero=True, **kwargs):
        """Initialize with a series of models

        model : 2D array of size (a, b, c, d, n)
            The model used for fitting.  a, b, c, d are the lengths of model
            parameters, n is the number of measurements.
        ti, emis, icedep, loss : 1D arrays of size (a,), (b,), (c,), (d,)
            Model parameters for thermal inertia, emissivity, icedepth, and
            loss tangent.
        lst : 1D array
            Local solar time array.  If provided, then data will be sorted
            by `lst`, and the dimensions will be labeled by it in plots.  If
            `None`, then data are not sorted.
        nonzero : bool, optional
            If `True`, only use non-zero values in the model for PCA and
            model fitting
        **kwargs : keyword parameters for sklearn.decomposition.PCA
        """
        # validate input
        sz = model.shape
        if (len(sz) != 5) or (sz[0] != len(ti)) or (sz[1] != len(emis)) \
            or ((sz[0], sz[2]) != icedep.shape) or (sz[3] != len(loss)):
            raise ValueError('Input parameter error.')
        # parss parameters
        self.data = model.copy()
        self._data1d = self.data.reshape(-1, sz[-1])
        self.ti = ti
        self.emissivity = emis
        self.icedepth = icedep
        self.losstangent = loss
        if lst is not None:
            self.lst = lst.copy()
        # filter out zero data point if needed
        if nonzero:
            self._nzi = np.abs(self._data1d).sum(axis=0) != 0
        else:
            self._nzi = np.ones(sz[-1], dtype=bool)
        ncomp = kwargs.pop('n_components', \
                                self._data1d[:, self._nzi].shape[1])
        # initialize PCA
        super().__init__(n_components=ncomp, **kwargs)
        self.data_transform = self.fit_transform(self._data1d[:, self._nzi])
        self.data_transform = self.data_transform.reshape(sz[:-1]+(ncomp,))
        self.data_transform1d = self.data_transform.reshape(-1, ncomp)

    def plot_eigenvector(self, i=None, ax=None):
        """Plot eigenvectors

        i : integer or integer iterables
            Index of eigen vector to be plotted.  If `None`, plot all
        ax : Axis
            Axis to plot.  If `None`, create a new axis
        """
        if not hasattr(i, '__iter__'):
            i = [i]
        if ax is None:
            ax = plt.figure().add_subplot(111)
        if hasattr(self, 'lst'):
            xx = self.lst[self._nzi]
            xlabel = 'Local Solar Time'
        else:
            xx = np.array(range(self.n_components))
            xlabel = 'Dimension'
        st = xx.argsort()
        for x in i:
            ax.plot(xx[st], self.components_[x][st],'-o')
        ax.set_xlabel(xlabel)
        plt.legend(['Eigen V{}'.format(x+1) for x in i])

    def plot_eigenvalues(self, ax=None, **kwargs):
        """Plot eigenvalues

        ax : Axis
            Axis to plot.  If `None`, create a new axis
        **kwargs : Keyword parameters accepted by pyplot.plot
        """
        if ax is None:
            ax = plt.figure().add_subplot(111)
        ax.plot(self.explained_variance_, '-o', **kwargs)
        ax.set_xlabel('Eigenvalue Index')

    @property
    def mean(self):
        return self.data_transform1d.mean(axis=0)

    @property
    def median(self):
        return np.median(self.data_transform1d, axis=0)

    @property
    def max(self):
        return self.data_transform1d.max(axis=0)

    @property
    def min(self):
        return self.data_transform1d.min(axis=0)


def in_hull(points, x):
    """Check if point `x` is in the convex hull of point cloud `points`

    points : numpy.ndarray of shape (m, n)
        Point clouds for m points in n-dimensional space
    x : numpy.ndarray of shape (n,)
        Point in n-dimensional space

    Algorithm from https://stackoverflow.com/a/43564754
    """
    from scipy.optimize import linprog
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


class PCAData():
    """PCA analysis for projected data
    """
    def __init__(self, data, lst, **kwargs):
        """Initialize with a series of models

        data : 3D array of size (n, i, j)
            Data projected into lon-lat projection.  n is number of
            observations (images), i is number of latitudes, and j is
            number of longitudes.
        lst : 3d array of size (n, i, j)
            Local solar time array.  Same shape as `data`.
        **kwargs : keyword parameters for sklearn.decomposition.PCA
        """
        self.data = data.copy()
        self.lst = lst.copy()
        self.pca = None

    def clustering(self, axis, lat, lst=np.linspace(9,17,20)):
        """Clustering analysis

        axis : [0, 'obs', 2, 'lon']
            The axis for clustering analysis.
            [0, 'obs'] means analysis along the observations direction.
            [2, 'lon'] means analysis along the longitudinal direction.
        lat : int
            Index along latitude direction

        Returns PCA class object.
        """
        from scipy.interpolate import interp1d
        sz = self.data.shape
        _data = self.data[:, lat, :]
        _lst = self.lst[:, lat, :]
        if axis in [0, 'obs']:
            pass
        elif axis in [2, 'lon']:
            if _data.ndim == 3:
                sz = _data.shape
                _data = _data.reshape(-1, sz[-1])
                _lst = _lst.reshape(-1, sz[-1])
            sz = _data.shape
            prof = []
            for i in range(sz[0]):
                ww = _data[i] > 0
                l = _lst[i, ww]
                t = _data[i, ww]
                s = l.argsort()
                l = l[s]
                t = t[s]
                prof.append(interp1d(l, t))
            f = np.array([p(lst) for p in prof])
            self.c_lst = lst  # common lst
            self.pca = PCA(n_components = len(lst))
            self.data_transform = self.pca.fit_transform(f)

    def plot_eigenvector(self, i=None, ax=None):
        """Plot eigenvectors

        i : integer or integer iterables
            Index of eigen vector to be plotted.  If `None`, plot all
        ax : Axis
            Axis to plot.  If `None`, create a new axis
        """
        if self.pca is None:
            raise ValueError('clustering analysis not performed yet, run '
                    '`.clustering(axis, lat)` first')
        if not hasattr(i, '__iter__'):
            i = [i]
        if ax is None:
            ax = plt.figure().add_subplot(111)
        for x in i:
            ax.plot(self.c_lst, self.pca.components_[x],'-o')
        ax.set_xlabel('Local Solar Time')
        plt.legend(['Eigen V{}'.format(x+1) for x in i])

    def plot_eigenvalues(self, ax=None, **kwargs):
        """Plot eigenvalues

        ax : Axis
            Axis to plot.  If `None`, create a new axis
        **kwargs : Keyword parameters accepted by pyplot.plot
        """
        if self.pca is None:
            raise ValueError('clustering analysis not performed yet, run '
                    '`.clustering(axis, lat)` first')
        if ax is None:
            ax = plt.figure().add_subplot(111)
        ax.plot(self.pca.explained_variance_, '-o', **kwargs)
        ax.set_xlabel('Eigenvalue Index')

    def mean(self):
        return self.data_transform1d.mean(axis=0)

    @property
    def median(self):
        return np.median(self.data_transform1d, axis=0)

    @property
    def max(self):
        return self.data_transform1d.max(axis=0)

    @property
    def min(self):
        return self.data_transform1d.min(axis=0)


class ALMACeresCentroid(ImageSet):
    """Base class for ALMA Ceres image centroiding"""

    @staticmethod
    def _ceres_image_loader(file):
        return np.squeeze(ALMACeresImage.from_fits(file))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loader = ALMACeresCentroid._ceres_image_loader
        self._xc = np.zeros(self._shape)
        self._yc = np.zeros(self._shape)
        self.attr.extend(['_xc', '_yc'])
        self._generate_flat_views()

    @property
    def center(self):
        return np.c_[self._yc, self._xc]


class EdgeDetectionCentroid(ALMACeresCentroid):
    """Find the centroid of Ceres by edge detection"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._smaj = np.zeros(self._shape)
        self._smin = np.zeros(self._shape)
        self._theta = np.zeros(self._shape)
        self._status = np.zeros(self._shape, dtype=bool)
        self.attr.extend(['_smaj', '_smin', '_theta', '_status'])
        self._generate_flat_views()

    def _scale2byte(self, im):
        return np.uint8((im - im.min()) / (im.max() - im.min()) * 255)

    def centroid_canny(self, index=None):
        from cv2 import Canny
        from skimage.measure import EllipseModel
        index_ = self._ravel_indices(index)
        for i in index_:
            if self.image is None or self._1d['image'][i] is None:
                self._load_image(i)
            im = self._scale2byte(self._1d['image'][i].copy())
            edge_found = False
            for aptsz in [3, 5, 7]:
                edge = Canny(im, 20, 100, apertureSize=3)
                if (edge > 10).any():
                    edge_found = True
                    break
            if not edge_found:
                continue
            on_edge = edge > 10
            yy, xx = np.indices(im.shape)
            pts = np.c_[xx[on_edge], yy[on_edge]]
            ell = EllipseModel()
            ell.estimate(pts)
            self._1d['_xc'][i], self._1d['_yc'][i], a, b, theta = ell.params
            if a < b:
                a, b = b, a
                theta = theta - np.pi/2
            self._1d['_smaj'][i] = a
            self._1d['_smin'][i] = b
            self._1d['_theta'][i] = np.rad2deg(theta)
            self._1d['_status'][i] = True

    def centroid_manual(self, index=None, threshold_mode='abs', threshold=100):
        from cv2 import GaussianBlur, Sobel, cartToPolar, CV_64F
        from skimage.measure import EllipseModel
        index_ = self._ravel_indices(index)
        for i in index_:
            if self.image is None or self._1d['image'][i] is None:
                self._load_image(i)
            im = self._scale2byte(self._1d['image'][i].copy())
            im_blur = GaussianBlur(im, (5, 5), 5)
            gx = Sobel(np.float32(im_blur), CV_64F, 1, 0, 3)
            gy = Sobel(np.float32(im_blur), CV_64F, 0, 1, 3)
            mag, ang = cartToPolar(gx, gy, angleInDegrees=True)
            if threshold_mode != 'abs':
                th = mag.max() * threshold
            else:
                th = threshold
            #print(mag.max(), th)
            on_edge = mag > th
            yy, xx = np.indices(im.shape)
            pts = np.c_[xx[on_edge], yy[on_edge]]
            ell = EllipseModel()
            ell.estimate(pts)
            #print(pts.shape)
            #print(ell.params)
            self._1d['_xc'][i], self._1d['_yc'][i], a, b, theta = ell.params
            if a < b:
                a, b = b, a
                theta = theta - np.pi/2
            self._1d['_smaj'][i] = a
            self._1d['_smin'][i] = b
            self._1d['_theta'][i] = np.rad2deg(theta)
            self._1d['_status'][i] = True


Centroid = EdgeDetectionCentroid  # for compatibility with old code


class CCCenterSearch():
    """Centroid the data with cross-correlation

    In the centroiding process, the model is shifted in a grid, and the
    cross-correlations between the model at each grid position and the
    corresponding patch in the data is calculated.  When the position of
    the model with the maximum cross-correlation is found, the model center
    in the pixel coordinate of the data is taken as the center of the data.
    """

    def __init__(self, data, model, center=None):
        """
        data, model : 2D arrays
            The data and corresponding model array.  If different shape,
            then the size of data must be no less than the size of model.
            The center of the model is assumed to be at pixel coordinate
            (model.shape - 1)/2, where the center of the lower-left pixel
            has a coordinate of (0, 0).
        center : (yc, xc), optional
            The center of data.  Default is at (data.shape -1 / 2).
        """
        self.data = data
        self.model = model
        if center is None:
            self.center = (np.array(data.shape) - 1) / 2
        else:
            self.center = center
        self.model_par = {}
        self.model_par['center'] = (np.array(self.model.shape) - 1) / 2
        self.model_par['integer_center'] = \
                            np.int64(np.round(self.model_par['center']))
        self.model_par['fractional_center'] = \
                self.model_par['center'] - self.model_par['integer_center']
        self.subim_par = {}

    def _cut_subimage(self, center):
        self.subim_par['center'] = center
        self.subim_par['integer_center'] = np.int64(np.round(center))
        self.subim_par['fractional_center'] = \
                            center - self.subim_par['integer_center']
        xx1 = self.subim_par['integer_center'] - \
                                        self.model_par['integer_center']
        xx2 = self.subim_par['integer_center'] + \
                (np.array(self.model.shape) - self.model_par['integer_center'])
        return self.data[xx1[0]:xx2[0], xx1[1]:xx2[1]]

    def _fractional_pixel_shift_model(self):
        fractioanl_shift = self.subim_par['fractional_center'] - \
                                self.model_par['fractional_center']
        return shift(self.model, fractioanl_shift)

    def search(self, stepsize=None, precision=0.01, maxiter=50,
            verbose=True):
        """Find the center of data

        Parameters
        ----------
        stepsize : None, number, or array of 2 numbers, optional
            Initial step size.  If `None`, then the default step size is 1/10
            of model size.  If a single number, then it's the step size in both
            directions.
        precision : number, optional
            Center search precision.  Iteration will stop if the precision
            is reached.
        maxiter : int, optional
            Maximum number of iteration.
        verbose : bool
        """
        from scipy.stats import pearsonr
        if stepsize is None:
            stepsize = np.array(self.model.shape) / 10
        if not hasattr(stepsize, '__iter__') or len(stepsize) == 1:
            stepsize = np.repeat(stepsize, 2)

        if verbose:
            print('Initial center: {:.4f}, {:.4f}'.format(self.center[0],
                        self.center[1]))
            print('Initial stepsize: {:.4f}, {:.4f}'.format(stepsize[0],
                        stepsize[1]))
            print('Precision: {:.4f}'.format(precision))
            print('Start iteration:')

        niter = 0
        change = np.max(self.model.shape)
        precision /= 2
        while ((np.max(change) > precision) \
                    or (np.max(stepsize) > precision)) \
                and (niter < maxiter):
            if verbose:
                print('  iteration {}'.format(niter+1))
            # calculate trial centers
            offset = np.outer(np.linspace(-5, 5, 11), stepsize)
            trial_centers = self.center + offset
            # calculate cross-correlation
            cc = np.zeros((11, 11))
            for i, yc in enumerate(trial_centers[:, 0]):
                for j, xc in enumerate(trial_centers[:, 1]):
                    subim = self._cut_subimage([yc, xc])
                    model_s = self._fractional_pixel_shift_model()
                    cc[i, j] = pearsonr(subim.flatten(), model_s.flatten())[0]
            y, x = np.unravel_index(cc.argmax(), cc.shape)
            center = trial_centers[[y, x], [0,1]]
            change = center - self.center
            self.center = center
            if verbose:
                print('    new center: {:.4f}, {:.4f}'.format(center[0],
                            center[1]))
                print('    changed by: {:.4f}, {:.4f}'.format(change[0],
                            change[1]))
                print('    stepsize: {:.4f}, {:.4f}'.format(stepsize[0],
                            stepsize[1]))
            if (y not in [0, 10]) and (change[0] < precision) \
                        and (stepsize[0] >= precision):
                stepsize[0] /= 2
            if (x not in [0, 10]) and (change[1] < precision) \
                        and (stepsize[1] >= precision):
                stepsize[1] /= 2
            niter += 1

        # calculate best match model and data
        self._matched_data = self._cut_subimage(self.center)
        self._matched_model = self._fractional_pixel_shift_model()
        self._cc = cc
        if verbose:
            print('Final center: {:.4f}, {:.4f}'.format(center[0], center[1]))

    def __call__(self, *args, **kwargs):
        self.search(*args, **kwargs)


class CCCentroid(ALMACeresCentroid):
    """Search for centers of images with cross-correlation"""

    def __init__(self, im, model, center=None, **kwargs):
        """
        Parameters
        ----------
        im : str array or ndarray
            Input image name or array.  See `jylipy.image.ImageSet`.
        model : ndarray
            Model used to match the image to search for center.  Same shape
            as `im`.
        center : array of shape (..., 2)
            Initial centers.
        **kwargs : other keywords accepted by `jylipy.image.ImageSet`.
        """
        super().__init__(im, **kwargs)
        self.model = np.zeros(self._shape, dtype='object')
        self._yc[:] = center[:, 0]
        self._xc[:] = center[:, 1]
        self._data = np.zeros(self._shape, dtype='object')
        self.attr.extend(['model', '_data'])
        self._generate_flat_views()
        model1d = model.reshape(-1, model.shape[-2], model.shape[-1])
        center1d = self.center.reshape(-1, 2)
        for i in range(self._size):
            if self.image is None or self._1d['image'][i] is None:
                self._load_image(i)
            ct = center1d[i]
            ct = None if (ct**2).sum() == 0 else ct
            self._1d['model'][i] = model1d[i]
            self._1d['_data'][i] = CCCenterSearch(self._1d['image'][i], \
                    self._1d['model'][i], center=ct)

    def centroid(self, extremely_verbose=False, **kwargs):
        verbose = kwargs.pop('verbose', True)
        if extremely_verbose:
            verbose = True
        for i in range(self._size):
            if verbose:
                print('image {}: '.format(i), end='')
            self._1d['_data'][i](verbose=extremely_verbose, **kwargs)
            self._1d['_yc'][i], self._1d['_xc'][i] = self._1d['_data'][i].center
            if verbose:
                print('{:.4f}, {:.4f}'.format(self._1d['_yc'][i],
                                              self._1d['_xc'][i]))

    def write(self, *args, **kwargs):
        self.attr.remove('model')
        self.attr.remove('_data')
        super().write(*args, **kwargs)
        self.attr.extend(['model', '_data'])


def cut_subimage(im, center, shape=(256, 256)):
    """Cut subimage

    im : `ALMACeresImage`
        Input image to cut
    center : (yc, xc)
        The center pixel coordinate of image
    shape : (ysize, xsize)
        The shape of subimage, must be even numbers

    Return
    ------
    Output subimage will be centered at (shape-1)/2, i.e., default
    at (127.5, 127.5), where the center of the lower-left pixel is
    (0, 0).
    """
    yc, xc = center
    sz = np.array(im.shape)
    ref = np.int64(np.round(sz/2))  # reference point
    if np.any(np.array(shape)//2 * 2 != np.array(shape)):
        raise ValueError('shape must be even numbers')
    outsz = np.array(shape)//2
    im_s = shift(im, (ref[0] + 0.5 - yc, ref[1] + 0.5 - xc))
    return im_s[ref[0]-outsz[0]+1:ref[0]+outsz[0]+1,
                ref[1]-outsz[1]+1:ref[1]+outsz[1]+1]
