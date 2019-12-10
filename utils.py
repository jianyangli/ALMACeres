"""Utility module to support ALMA Ceres data analysis
"""

import numpy as np
import numbers, inspect, json
from os import path
from astropy.io import fits
from astropy import units, table, nddata, time
import astropy.units as u
import spiceypy as spice
from .vector import vecsep, sph2xyz


def is_iterable(v):
    """Check whether a variable is iterable"""
    if isinstance(v, (str, bytes)):
        return False
    elif hasattr(v, '__iter__'):
        return True
    else:
        return False


def ascii_read(*args, **kwargs):
    '''astropy.io.ascii.read wrapper

    Same API and functionalities as ascii.read
    Returns the extended Table class'''
    return table.Table(ascii.read(*args, **kwargs))


def headfits(imfile, ext=0, verbose=True):
    '''IDL headfits emulator.

 Parameters
 ----------
 imfile : string, or list of string
   FITS file name(s)
 ext : non-negative integer, optional
   The extension to be read in
 verbose : bool, optional
   Suppress the screen print of FITS information if False

 Returns
 -------
 astropy.io.fits.header.Header instance, or list of it

 v1.0.0 : JYL @PSI, Nov 17, 2013
    '''

    if is_iterable(imfile):
        return [headfits(f,ext=ext,verbose=verbose) for f in imfile]
    elif not isinstance(imfile, (str, bytes)):
        raise ValueError('string types or iterable expected, {0} received'.format(type(imfile)))

    fitsfile = fits.open(imfile)
    if verbose:
        fitsfile.info()

    if ext >= len(fitsfile):
        print()
        print(('Error: Extension '+repr(ext)+' does not exist!'))
        return None

    if fitsfile[ext].data is None:
        print()
        print(('Error: Extension '+repr(ext)+' contains no image!'))
        return None

    return fitsfile[ext].header


def readfits(imfile, ext=0, verbose=True, header=False):
    '''IDL readfits emulator.

 Parameters
 ----------
 imfile : string, or list of strings
   FITS file name(s)
 ext : non-negative integer, optional
   The extension to be read in
 verbose : bool, optional
   Suppress the screen print of FITS information if False
 header : bool, optional
   If `True`, then (image, header) tuple will be returned

 Returns
 -------
 image, or tuple : (image, header)
   image : ndarray or list of ndarray of float32
   header : astropy.io.fits.header.Header instance, or list of it

 v1.0.0 : JYL @PSI, Nov 17, 2013
 v1.0.1 : JYL @PSI, 5/26/2015
   Accept extension name for `ext`.
   Return the actual header instead of `None` even if extension
     contains no data
   Returned data retains the original data type in fits.
    '''

    if isinstance(imfile, (str,bytes)):
        fitsfile = fits.open(imfile)
        if verbose:
            fitsfile.info()

        try:
            extindex = fitsfile.index_of(ext)
        except KeyError:
            print()
            print(('Error: Extension {0} not found'.format(ext)))
            if header:
                return None, None
            else:
                return None

        if extindex >= len(fitsfile):
            print()
            print(('Error: Requested extension number {0} does not exist'.format(extindex)))
            img, hdr = None, None
        else:
            hdr = fitsfile[extindex].header
            if fitsfile[extindex].data is None:
                print()
                print(('Error: Extension {0} contains no image'.format(ext)))
                img = None
            else:
                img = fitsfile[extindex].data

        if header:
            return img, hdr
        else:
            return img
    elif hasattr(imfile,'__iter__'):

        img = [readfits(f, ext=ext, verbose=verbose) for f in imfile]
        if header:
            return img, headfits(imfile, ext=ext, verbose=verbose)
        else:
            return img

    else:
        raise TypeError('str or list of str expected, {0} received'.format(type(imfile)))


def writefits(imfile, data=None, header=None, name=None, append=False, overwrite=False):
    '''IDL writefits emulator'''
    if append:
        hdu = fits.ImageHDU(data, header=header, name=name)
        hdulist = fits.open(imfile)
        hdulist.append(hdu)
        hdulist.writeto(imfile, overwrite=True)
    else:
        hdu = fits.PrimaryHDU(data, header=header)
        hdu.writeto(imfile, overwrite=overwrite)


def makenxy(y1, y2, ny, x1, x2, nx, rot=None):
    '''Make 2-d y and x coordinate arrays of specified dimensions
  (Like IDL JHU/APL makenxy.pro)

 Parameters
 ----------
 y1, y2 : float
   Min and max coordinate of the first dimension in output array
 ny : float
   Number of steps in the first dimension
 x1, x2 : float
   Min and max coordinate of the second dimension in output array
 nx : float
   Number of steps in the second dimension
 rot : float
   Rotation of arrays

 Returns
 -------
 yarray, xarray : 2-D arrays

 v1.0.0 : JYL @PSI, June 2, 2014
    '''

    y, x = np.indices((ny,nx), float)
    y = y*(y2-y1)/(ny-1)+y1
    x = x*(x2-x1)/(nx-1)+x1

    if rot is not None:
        m = rotm(rot)
        x, y = m[0,0]*x+m[0,1]*y, m[1,0]*x+m[1,1]*y

    return np.array((y, x))


def resmean(x, threshold=3., std=False):
    '''Calculate resistant mean.

 Parameters
 ----------
 x : array-like, numbers
 cut : float, optional
   Threshold for resistant mean
 std : bool, optional
   If set True, then a tuple of (mean, std_dev) will be returned

 Returns
 -------
 float, or tuple of two float

 v1.0.0 : JYL @PSI, Oct 24, 2013
    '''

    x1 = np.asarray(x).flatten()

    while True:
        m, s = x1.mean(), x1.std()
        ww = (x1 <= m+threshold*s) & (x1 >= m-threshold*s)
        x1 = x1[ww]
        if np.abs(x1.std()-s) < 1e-13:
            break

    if std:
        return x1.mean(), x1.std()
    else:
        return x1.mean()


import contextlib
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def is_iterable(v):
    """Check whether a variable is iterable"""
    if isinstance(v, (str, bytes)):
        return False
    elif hasattr(v, '__iter__'):
        return True
    else:
        return False


def aperture_photometry(data, apertures, **kwargs):
    '''Wrapper for photutils.aperture_photometry

    Fixes a bug in the original program (see below).
    Returns the extended Table class.

    ...bug description:
    The original program that puts the `xcenter` and `ycenter` in a
    shape (1,1) array when input aperture contains only one position.

    v1.0.0 : JYL @PSI, Feb 2015'''

    from photutils import aperture_photometry
    ap = table.Table(aperture_photometry(data, apertures, **kwargs))
    if apertures.positions.shape[0] == 1:
        xc = ap['xcenter'].data[0]
        yc = ap['ycenter'].data[0]
        ap.remove_column('xcenter')
        ap.remove_column('ycenter')
        ap.add_column(table.Column([xc],name='xcenter'))
        ap.add_column(table.Column([yc],name='ycenter'))
    if ap['xcenter'].unit is None:
        ap['xcenter'].unit = units.pix
    if ap['ycenter'].unit is None:
        ap['ycenter'].unit = units.pix
    return ap


def apphot(im=None, aperture=None, ds9=None, radius=3, newwindow=False, **kwargs):
    '''Measure aperture photometry

    If no image is given, then the image will be extracted from the
    active frame in a DS9 window specified by `ds9.`

    If no DS9 window is specified, then the first-openned DS9 window
    will be used.

    If no aperture is given, then the aperture(s) will be extracted
    from the circular and annulus regions in the DS9 window.  If no
    regions is defined, then the aperture center will be specified
    interactively by mouse click in DS9, and the radius of apertures
    will be specified by keyword `radius`.

    Photometry will be returned in a Table.

    v1.0.0 : JYL @PSI, Feb 2015
    v1.0.1 : JYL @PSI, Jan 18, 2017
      Bug fix
    '''
    if im is None and ds9 is None:
        raise ValueError('either `im` or `ds9` must be specified')

    if im is None:  # Get image from a ds9 window
        im = getds9(ds9).get_arr2np()

    if aperture is None:  # If no aperture specified, then get it from DS9
        ds9 = getds9(ds9, newwindow)
        ds9.imdisp(im)
        aperture = ds9.aperture()
        if not aperture:
            centroid = kwargs.pop('centroid', False)
            aperture = ds9.define_aperture(radius=radius, centroid=centroid)
    if aperture == []:  # If still not specified, do nothing
        return None

    # Measure photometry
    pho, r, r1 = [], [], []
    if not isinstance(aperture,list):
        aperture = [aperture]
    for apt in aperture:
        napt = apt.positions.shape[0]
        if hasattr(im,'uncertainty'):
            error = im.uncertainty.array
        else:
            error = None
        error = kwargs.pop('error', error)
        pho.append(aperture_photometry(im, apt, error=error, **kwargs))
        if napt == 1:
            r.append(getattr(apt, 'r', getattr(apt, 'r_in', None)))
            r1.append(getattr(apt, 'r_out', None))
        else:
            r = r+[getattr(apt, 'r', getattr(apt, 'r_in', None))]*napt
            r1 = r1+[getattr(apt, 'r_out', None)]*napt
    if pho == []:
        return None
    pho = table.Table(table.vstack(pho))
    pho.add_column(table.Column(r, name='r', unit='pix'))
    if len(np.nonzero(r1)[0]) > 0:
        pho.add_column(table.Column(r1, name='r_out', unit='pix'))
    return pho


def subcoord(time, target, observer='earth', bodyframe=None, saveto=None, planetographic=False):
    '''
 Calculate the sub-observer and sub-solar points.

 Parameters
 ----------
 time : str, array-like
   Time(s) to be calculated.  Must be in a format that can be accepted
   by SPICE.
 target : str
   The name of target that SPICE accepts.
 observer : str, optional
   The name of observer that SPICE accepts.
 bodyframe : str, optional
   The name of the body-fixed coordinate system, if not standard (in
   the form of target+'_fixed')
 saveto : str, file, optional
   Output file
 planetographic : bool, optional
   If `True`, then the planetographic coordinates are returned as
   opposed to planetocentric.

 Return
 ------
 Astropy Table:
   'sslat' : sub-solar latitude
   'sslon' : sub-solar longitude
   'solat' : sub-observer latitude
   'solon' : sub-observer latitude
   'rh' : heliocentric distance
   'range' : observer distance
   'phase' : phase angle
   'polepa' : position angle of pole
   'poleinc' : inclination of pole
   'sunpa' : position angle of the Sun
   'suninc' : inclination angle of the Sun
 Position angles are measured from celestial N towards E.  inclination
 angles are measured from sky plane (0 deg) towards observer (+90).
 All angles are in deg, all distances are in AU

 v1.0.0 : JYL @PSI, May 23, 2014
 v1.0.1 : JYL @PSI, July 8, 2014
   Modified the call to spice.gdpool to accomodate a behavior that is
   different from what I remember
   Modified the call to spice.str2et
 v1.0.2 : JYL @PSI, July 15, 2014
   Changed the calculation of pole orientation to accomodate the case
   where the frame is expressed in the kernel pool differently than
   for small bodies.
 v1.0.3 : JYL @PSI, October 8, 2014
   Fixed the bug when input time is a scalor
   Change return to an astropy Table
 v1.0.4 : JYL @PSI, November 19, 2014
   Small bug fix for the case when input time is a scalor
   Small bug fix for the output table unit and format
 v1.0.5 : JYL @PSI, October 14, 2015
   Add keyword `planetographic`
   Add target RA and Dec in the return table
   Increased robustness for the case when no body frame is defined
   Change the table headers to start with capital letters
   Improve the program structure
    '''

    # Determine whether kernel pool for body frame exists
    if bodyframe is None:
        bodyframe = target+'_fixed'
    try:
        kp = spice.gipool('FRAME_'+bodyframe.upper(),0,1)
    except spice.utils.support_types.SpiceyError:
        kp = None
    if kp is not None:
        code = kp[0]
        polera = spice.bodvrd(target, 'POLE_RA', 3)[1][0]
        poledec = spice.bodvrd(target, 'POLE_DEC', 3)[1][0]
        r_a, r_b, r_c = spice.bodvrd(target, 'RADII', 3)[1]
        r_e = (r_a+r_b)/2
        flt = (r_e-r_c)/r_e

    # Process input time
    if isinstance(time,str):
        et = [spice.str2et(time)]
        time = [time]
    elif (not isinstance(time, (str,bytes))) and hasattr(time, '__iter__'):
        et = [spice.str2et(x) for x in time]
    else:
        raise TypeError('str or list of str expected, {0} received'.format(type(time)))

    # Prepare for iteration
    sslat, sslon, solat, solon, rh, delta, phase, polepa, poleinc, sunpa, suninc, tgtra, tgtdec = [], [], [], [], [], [], [], [], [], [], [], [], []
    if kp is None:
        workframe = 'J2000'
    else:
        workframe = bodyframe

    # Iterate over time
    for t in et:

        # Target position (r, RA, Dec)
        pos1, lt1 = spice.spkpos(target, t, 'J2000', 'lt+s', observer)
        pos1 = np.array(pos1)
        rr, ra, dec = spice.recrad(pos1)
        delta.append(rr*units.km.to(units.au))
        tgtdec.append(np.rad2deg(dec))
        tgtra.append(np.rad2deg(ra))

        # Heliocentric distance
        pos2, lt2 = spice.spkpos('sun', t-lt1, 'J2000', 'lt+s', target)
        from numpy.linalg import norm
        rh.append(norm(pos2)*units.km.to('au'))

        # Phase angle
        phase.append(vecsep(-pos1, pos2, directional=False))

        # Sun angle
        m = np.array(spice.twovec(-pos1, 3, [0,0,1.], 1))
        rr, lon, lat = spice.recrad(m.dot(pos2))
        sunpa.append(np.rad2deg(lon))
        suninc.append(np.rad2deg(lat))

        if kp is not None:

            # Sub-observer point
            pos1, lt1 = spice.spkpos(target, t, bodyframe, 'lt+s', observer)
            pos1 = np.array(pos1)
            if planetographic:
                lon, lat, alt = spice.recpgr(target, -pos1, r_e, flt)
            else:
                rr, lon, lat = spice.recrad(-pos1)
            solat.append(np.rad2deg(lat))
            solon.append(np.rad2deg(lon))

            # Sub-solar point
            pos2, lt2 = spice.spkpos('sun', t-lt1, bodyframe, 'lt+s', target)
            lon, lat, alt = spice.recpgr(target, pos2, r_e, 0.9)
            #print np.rad2deg(lon), np.rad2deg(lat)
            rr, lon, lat = spice.recrad(pos2)
            #print np.rad2deg(lon), np.rad2deg(lat)
            if planetographic:
                lon, lat, alt = spice.recpgr(target, pos2, r_e, flt)
            else:
                rr, lon, lat = spice.recrad(pos2)
            sslon.append(np.rad2deg(lon))
            sslat.append(np.rad2deg(lat))

            # North pole angle
            pole = [polera, poledec]
            rr, lon, lat = spice.recrad(m.dot(sph2xyz(pole)))
            polepa.append(np.rad2deg(lon))
            poleinc.append(np.rad2deg(lat))

    if kp is None:
        tbl = table.Table((time, rh, delta, phase, tgtra, tgtdec), names='Time rh Range Phase RA Dec'.split())
    else:
        tbl = table.Table((time, rh, delta, phase, tgtra, tgtdec, solat, solon, sslat, sslon, polepa, poleinc, sunpa, suninc), names='Time rh Range Phase RA Dec SOLat SOLon SSLat SSLon PolePA PoleInc SunPA SunInc'.split())

    for c in tbl.colnames[1:]:
        tbl[c].format='%.2f'
        tbl[c].unit = units.deg
    tbl['Time'].format='%s'
    tbl['Time'].unit=None
    tbl['rh'].format = '%.4f'
    tbl['rh'].unit = units.au
    tbl['Range'].format = '%.4f'
    tbl['Range'].unit = units.au
    if saveto is not None:
        tbl.write(saveto)

    return tbl


def obj2dict(obj, exclude=['_','__']):
    """Return all properties of input class instance in a dictionary
    """
    d = dict(
        (key, value)
        for key, value in inspect.getmembers(obj)
        if not inspect.isabstract(value)
        and not inspect.isbuiltin(value)
        and not inspect.isfunction(value)
        and not inspect.isgenerator(value)
        and not inspect.isgeneratorfunction(value)
        and not inspect.ismethod(value)
        and not inspect.ismethoddescriptor(value)
        and not inspect.isroutine(value)
    )
    if len(exclude) > 0:
        for p in list(d.keys()):
            for e in exclude:
                if p.startswith(e):
                    d.pop(p)
                    break
    return d


class ObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return self.default(obj.to_json())
        elif hasattr(obj, '__dict__'):
            d = obj2dict(obj)
            return self.default(d)
        else:
            return obj


class JSONSerializable():
    """A base class to serialize class objects
    """
    def __init__(self, from_string=None, from_file=None, **kwargs):
        if from_string is not None or from_file is not None:
            self.from_json(string=from_string, file=from_file, **kwargs)

    def __str__(self):
        return str(obj2dict(self))

    def __repr__(self):
        return str(obj2dict(self))

    @classmethod
    def from_json(cls, string=None, file=None, **kwargs):
        """
        Load meta data from JSON string or file
        """
        if string is None and file is None:
            raise IOError('input string or file not supplied')
        if string is not None:
            s = json.loads(string, **kwargs)
        elif file is not None:
            with open(file, 'r') as f:
                s = json.load(f, **kwargs)
                f.close()
        obj = cls()
        for k, v in s.items():
            setattr(obj, k, v)
        return obj

    def to_json(self, saveto=None, **kwargs):
        """Dump meta data to a JSON string or file
        """
        if saveto is not None:
            indent = kwargs.pop('indent', 4)
            with open(saveto, 'w') as f:
                json.dump(self, f, default=obj2dict, indent=indent, **kwargs)
                f.close()
        else:
           return obj2dict(self)


class MetaData(JSONSerializable):
    """
    Meta data class for imaging and spectroscopic data
    """
    def __init__(self, datafile=None, **kwargs):
        super().__init__(**kwargs)
        if datafile is not None:
            self.collect(datafile, **kwargs)

    def collect(self, datafile):
        """
        Collect meta data from data file
        """
        self.path = path.dirname(datafile)
        self.name = path.basename(datafile)


class MetaDataList(list):
    """List of image meta data
    """
    def __init__(self, cls, **kwargs):
        """
        cls : name of `MetaData` or its subclass
        """
        super().__init__(**kwargs)
        self.member_class = cls

    def collect(self, datafiles, **kwargs):
        pars = []
        for i in range(len(datafiles)):
            p = {}
            if len(kwargs) > 0:
                for k in kwargs.keys():
                    p[k] = kwargs[k][i]
            pars.append(p)
        for f, p in zip(datafiles, pars):
            meta = self.member_class()
            meta.collect(f, **p)
            self.append(meta)

    def to_json(self, saveto=None, **kwargs):
        """Dump meta data collection to a JSON string or file
        """
        indent = kwargs.pop('indent', 4)
        out = [obj2dict(m) for m in self]
        for x in out:
            for k in x.keys():
                if isinstance(x[k], tuple(np.typeDict[k] for k in \
                                            np.typeDict.keys())):
                    pass
                elif isinstance(x[k], str):
                    pass
                elif isinstance(x[k], u.Quantity):
                    x[k] = x[k].value
                elif isinstance(x[k], time.Time):
                    x[k] = x[k].utc.isot
                else:
                    x[k] = 'Not JSON serializable'
        if saveto is not None:
            with open(saveto, 'w') as f:
                json.dump(out, f, indent=indent, **kwargs)
                f.close()
        else:
            return json.dumps(out, indent=indent, **kwargs)

    def to_table(self, saveto=None, overwrite=False):
        """Return meta data as an astropy Table class object
        """
        if len(self) == 0:
            return None
        nn = len(self)
        out = {}
        for k in obj2dict(self[0]).keys():
            out[k] = [getattr(m, k) for m in self]
            if isinstance(out[k][0], u.Quantity):
                out[k] = u.Quantity(out[k])
            elif isinstance(out[k][0], np.str_):
                out[k] = [str(x) for x in out[k]]
            elif isinstance(out[k][0], time.Time):
                out[k] = [x.utc.isot for x in out[k]]
            elif isinstance(out[k][0], list):
                out[k] = ['object'] * len(out[k])
        out = table.Table(out)
        if saveto is not None:
            out.write(saveto, overwrite=overwrite)
        else:
            return out

    @classmethod
    def from_json(cls, member_class, string=None, file=None, **kwargs):
        """Initialize class object from JSON strong or file

        member_class: class name of list members
        string: optional, string, the JSON string to initialize members.
            `string` parameter overrides `file` parameter
        file: optional, string, the name of JSON file
        """
        if string is not None:
            s = json.loads(string, **kwargs)
        elif file is not None:
            with open(file, 'r') as f:
                s = json.load(f, **kwargs)
                f.close()
        else:
            raise IOError('input string or file not supplied')
        obj = cls(member_class)
        for i in s:
            member = member_class()
            for k, v in i.items():
                setattr(member, k, v)
            obj.append(member)
        return obj

    @classmethod
    def from_table(cls, member_class, table=None, file=None):
        """Initialize object from astropy Table or table file

        member_class: class name of list members
        """
        if table is not None:
            if not isinstance(table, astropy.table.Table):
                raise ValueError('input table is not an astropy Table class object')
        elif file is not None:
            from astropy.io import ascii
            table = ascii.read(file)
        else:
            raise IOError('input table or file not supplied')
        obj = cls(member_class)
        keys = list(table.keys())
        for row in table:
            member = member_class()
            for k in keys:
                setattr(member, k, row[k])
            obj.append(member)
        return obj


def centroid(im, center=None, error=None, mask=None, method=0, box=6, tol=0.01, maxiter=50, threshold=None, verbose=False):
    '''Wrapper for photutils.centroiding functions

    Parameters
    ----------
    im : array-like, astropy.nddata.NDData or subclass
      Input image
    center : (y, x), optional
      Preliminary center to start the search
    error : array-like, optional
      Error of the input image.  If `im` is NDData type, then `error` will
      be extracted from NDData.uncertainty.  This keyword overrides the
      uncertainty in NDData.
    mask : array-like bool, optional
      Mask of input image.  If `im` is NDData type, then `mask` will be
      extracted from NDData.mask.  This keyword overrides the mask in NDData.
    method : int or str, optional
      Method of centroiding:
      [0, '2dg', 'gaussian'] - 2-D Gaussian
      [1, 'com'] - Center of mass
      [2, 'geom', 'geometric'] - Geometric center
    box : int, optional
      Box size for the search
    tol : float, optional
      The tolerance in pixels of the center.  Program exits iteration when
      new center differs from the previous iteration less than `tol` or number
      of iteration reaches `maxiter`.
    maxiter : int, optional
      The maximum number of iterations in the search
    threshold : number, optional
      Threshold, only used for method=2
    verbose : bool, optional
      Print out information

    Returns
    -------
    (y, x) as a numpy array

    This program uses photutils.centroids.centroid_2dg() or .centroid_com()

    v1.0.0 : JYL @PSI, Feb 19, 2015
    '''
    from photutils.centroids import centroid_2dg, centroid_com
    if isinstance(im, nddata.NDData):
        if error is None:
            if im.uncertainty is not None:
                error = im.uncertainty.array
        if mask is None:
            if im.mask is not None:
                mask = im.mask
        im = im.data

    if center is None:
        center = np.asarray(im.shape)/2.
    else:
        center = np.asarray(center)
    b2 = box/2
    if (method in [2, 'geom', 'geometric']) and (threshold is None):
        raise ValueError('threshold is not specified')
    if verbose:
        print(('Image provided as a '+str(type(im))+', shape = ', im.shape))
        print(('Centroiding image in {0}x{0} box around ({1},{2})'.format(box,center[0],center[1])))
        print(('Error array '+condition(error is None, 'not ', ' ')+'provided'))
        print(('Mask array '+condition(mask is None, 'not ', ' ')+'provided'))
    i = 0
    delta_center = np.array([1e5,1e5])
    while (i < maxiter) and (delta_center.max() > tol):
        if verbose:
            print(('  iteration {0}, center = ({1},{2})'.format(i, center[0], center[1])))
        p1, p2 = np.floor(center-b2).astype('int'), np.ceil(center+b2).astype('int')
        subim = np.asarray(im[p1[0]:p2[0],p1[1]:p2[1]])
        if error is None:
            suberr = None
        else:
            suberr = np.asarray(error[p1[0]:p2[0],p1[1]:p2[1]])
        if mask is None:
            submask = None
        else:
            submask = np.asarray(mask[p1[0]:p2[0],p1[1]:p2[1]])
        if method in [0, '2dg', 'gaussian']:
            xc, yc = centroid_2dg(subim, error=suberr, mask=submask)
        elif method in [1, 'com']:
            xc, yc = centroid_com(subim, mask=submask)
        elif method in [2, 'geom', 'geometric']:
            xc, yc = geometric_center(subim, threshold, mask=submask)
        else:
            raise ValueError("unrecognized `method` {0} received.  Should be [0, '2dg', 'gaussian'] or [1, 'com']".format(method))
        center1 = np.asarray([yc+p1[0], xc+p1[1]])
        delta_center = abs(center1-center)
        center = center1
        i += 1

    if verbose:
        print(('centroid = ({0},{1})'.format(center[0],center[1])))
    return center



import pyds9
class DS9(pyds9.DS9):
    '''Extended pyds9.DS9 class.'''

    def __init__(self, restore=None, **kwargs):
        super(DS9, self).__init__(**kwargs)
        if restore is not None:
            from os.path import isfile
            if not isfile(restore):
                raise Warning('restoration file '+restore+' not found')
            else:
                self.restore(restore)

    @property
    def frames(self):
        return self.get('frame all').split()

    @property
    def actives(self):
        return self.get('frame active').split()

    def cursor(self, coord='image', value=False):
        '''Return cursor position (y, x) in 0-based indices

        x, y = cursor()'''
        x, y = self.get('imexam coordinate '+coord).split()
        if value:
            return float(x)-1, float(y)-1, float(self.get(' '.join(['data', coord, x, y, '1 1 yes'])))
        else:
            return float(x)-1, float(y)-1

    def get_arr2np(self):
        '''Replacement of the original pyds9.DS9.get_arr2np(), which seems
        to return a float32 array with bytes swapped, and the image size
        corrected.'''
        im = super(DS9, self).get_arr2np().byteswap()
        return im.reshape(*im.shape[::-1])

    def xpa(self):
        '''Interactive XPA command session

        Example:

        >>> Enter XPA command: get frame #  # print current frame number
        >>> Enter XPA command: set frame next  # set next frame active
        >>> Enter XPA command: quit   # or q, quick XPA session'''
        import sys
        while True:
            sys.stdout.write('>>> Enter XPA command: ')
            xpa = sys.stdin.readline().strip('\n')#('XPA command: ')
            if xpa in ['quit', 'q']:
                break
            elif xpa.startswith('get'):
                cmd = xpa[xpa.find(' ')+1:]
                try:
                    print((self.get(cmd)))
                except:
                    print('Invalid XPA command')
            elif xpa.startswith('set'):
                cmd = xpa[xpa.find(' ')+1:]
                try:
                    r = self.set(cmd)
                except:
                    print('Invalid XPA command')
                if r != 1:
                    print(('Error in executing XPA command "'+xpa+'"'))
            else:
                print('Invalid XPA command')

    def _collect_pars(self):
        fno0 = self.get('frame')
        fnos = []
        if not hasattr(self, 'data'):
            self.data = {}
        self.set('frame first')
        while True:
            n = self.get('frame')
            if n in fnos:
                break
            fnos.append(n)
            if not hasattr(self.data, n):
                self.data[n] = {}
            self.data[n]['data'] = self.get_arr2np()
            self.data[n]['shift'] = [0.,0.]
            self.data[n]['rotate'] = 0.
            self.set('frame next')
        self.set('frame '+str(fno0))

    def interactive(self):
        '''Start an interactive session

        Commands:
          c : create new frame
          d : delete current frame
          f : zoom to fit
          option h : open header dialog window
          i : zoom in by a factor of 2
          m : match image coordinate, scale, and colorbar with current frame
          n : next frame
          o : zoom out by a factor of 2
          p : previous frame
          option p : open pan zoom rotate dialog window
          q : quit interactive session
          r : rotate image 1 deg in ccw direction
          shift r : rotate image 1 deg in cw direction
          option s : open scale dialog window
          shift x : XPA command window in Python
          arrow keys : shift image by 1 pixel
        '''
        self._collect_pars()
        shift = False
        option = False
        while True:
            k = self.get('imexam any coordinate image').split()[0]
            if k.startswith('Shift'):
                shift = True
            elif k.startswith('Mode'):
                option = True
            elif k == 'c':
                self.set('frame new')
            elif k == 'd':
                self.set('frame delete')
            elif k == 'f':
                self.set('zoom to fit')
            elif option and k == 'h':
                self.set('header')
                option = False
            elif k == 'i':
                self.set('zoom 2')
            elif k == 'm':
                self.set('match frame image')
                self.set('match scale')
                self.set('match colorbar')
            elif k == 'n':
                self.set('frame next')
            elif k == 'o':
                self.set('zoom 0.5')
            elif not option and k == 'p':
                self.set('frame prev')
                option = False
            elif option and k == 'p':
                self.set('pan open')
                option = False
            elif k == 'q':
                break
            elif not shift and k == 'r':
                self.set('rotate +1')
                self.data[self.get('frame')]['rotate'] += 1
            elif shift and k == 'r':
                self.set('rotate -1')
                self.data[self.get('frame')]['rotate'] -= 1
                shift = False
            elif option and k == 's':
                self.set('scale open')
                option = False
            elif shift and k == 'x':
                self.xpa()
                shift = False
            elif k == 'Right':
                self.set_np2arr(shift(self.get_arr2np(),(0,1)))
                self.data[self.get('frame')]['shift'][1] += 1
            elif k == 'Left':
                self.set_np2arr(shift(self.get_arr2np(),(0,-1)))
                self.data[self.get('frame')]['shift'][1] -= 1
            elif k == 'Up':
                self.set_np2arr(shift(self.get_arr2np(),(1,0)))
                self.data[self.get('frame')]['shift'][0] += 1
            elif k == 'Down':
                self.set_np2arr(shift(self.get_arr2np(),(-1,0)))
                self.data[self.get('frame')]['shift'][0] -= 1

    def imdisp(self, im, ext=None, par=None, newframe=True, verbose=True):
        '''Display images.

        Parameters
        ----------
        im : string or string sequence, 2-D or 3-D array-like numbers
          File name, sequence of file names, image, or stack of images.  For
          3-D array-like input, the first dimension is the dimension of stack
        ext : non-negative integer, optional
          The extension to be displayed
        newframe : bool, optional
          If set `False`, then the image will be displayed in the currently
          active frame in DS9, and the previous image will be overwritten.
          By default, a new frame will be created to display the image.
        par : string, or list of string, optional
          The display parameters for DS9.  See DS9 document.
        verbose : bool, optional
          If `False`, then all print out is suppressed.

        Returns: int or list(int)
          The status code:
          0 - no error
          1 - image file not found
          2 - extension not existent
          3 - invalid image extension
          4 - invalid PDS format
          5 - invalid FITS format
          6 - Unrecognized FITS extension
          7 - Unrecognized FITS extension error

        v1.0.0 : JYL @PSI, 2/14/2015, adopted from the standalone imdisp()
        '''

        # Pre-process for the case of a single image
        from astropy import nddata
        if isinstance(im, (str,bytes)):
            ims = [im]
        elif isinstance(im, np.ndarray):
            if im.ndim == 2:
                ims = [im]
            else:
                ims = im
        elif isinstance(im, nddata.NDData):
            ims = [im]
        else:
            ims = im

        # Loop through all images
        if len(ims) > 1:
            self.set('tile')
        st = []
        for im in ims:
            if newframe:
                self.set('frame new')

            # Display image(s)
            if isinstance(im, (str,bytes)):
                from os.path import isfile
                if not isfile(im):
                    if verbose:
                        print()
                        print('File does not exist: {0}'.format(im))
                    st.append(1)
                elif im.split('[')[0].lower().endswith(('.fits','.fit','fz')):
                    try:
                        if ext is None:
                            tmp = self.set('fits {0}'.format(im))
                        else:
                            tmp = self.set('fits {0}[{1}]'.format(im,ext))
                        st.append(0)
                    except ValueError:
                        if ext is None:
                            if verbose:
                                print()
                                print('Invalid FITS format')
                            st.append(5)
                        else:
                            from astropy.io import fits
                            info = fits.info(im,output=False)
                            if ext >= len(info):
                                if verbose:
                                    print()
                                    print('Error: Extension '+repr(ext)+' does not exist!')
                                st.append(2)
                            elif (info[ext][3] in ('ImageHDU','CompImageHDU')) and (len(info[ext][5])>1):
                                if verbose:
                                    print()
                                    print('Error: Extension '+repr(ext)+' contains no image!')
                                    print()
                                st.append(3)
                            else:
                                if verbose:
                                    print()
                                    print('Unrecognized FITS extension error')
                                st.append(7)
                            print((fits.info(im)))
                elif im.lower().endswith('.img'):
                    from .PDS import readpds
                    try:
                        self.set_np2arr(np.asarray(readpds(im)).astype('f4'))
                        st.append(0)
                    except:
                        if verbose:
                            print()
                            print('Invalid PDS format')
                        st.append(4)
                else:
                    if verbose:
                        print()
                        print('Unrecognized extension')
                    st.append(6)
            else:
                self.set_np2arr(np.asarray(im).astype('f4'))
                st.append(0)

            # set DS9 parameters
            if st[-1] == 0:
                if par is not None:
                    self.sets(par)
            else:
                self.set('frame delete')

        if len(st) == 1:
            st = st[0]
        return st

    def multiframe(self, fitsfile):
        '''Display multiframe FITS'''
        self.set('multiframe '+fitsfile)

    def region(self, frame=None, system='image', zerobased=True):
        '''Returns a list of regions already defined in the frame

        Note: the keyword `zerobased` controls the coordinate indexing
        convention.  DS9 convention is 1-based, but Python convention
        is 0-based!'''
        if frame is not None:
            fno0 = self.get('frame')
            self.set('frame '+str(frame))
        cf = self.get('frame')
        sys0 = self.get('region system')
        if sys0 != system:
            self.set('region system '+system)
        regstr = self.get('region -format saoimage')
        if regstr == []:
            return []
        else:
            regstr = regstr.split()
        reg = []
        for r in regstr:
            shape = r[:r.find('(')]
            pars = np.fromstring(r[r.find('('):].strip('()'),sep=',',dtype=float)
            if zerobased:
                pars[:2] -= 1
            if shape == 'circle':
                reg.append(CircularRegion(*pars, ds9=self, frame=cf, zerobased=zerobased))
            elif shape == 'ellipse':
                reg.append(EllipseRegion(*pars, ds9=self, frame=cf, zerobased=zerobased))
            elif shape == 'box':
                reg.append(BoxRegion(*pars, ds9=self, frame=cf, zerobased=zerobased))
            elif shape == 'annulus':
                reg.append(AnnulusRegion(*pars, ds9=self, frame=cf, zerobased=zerobased))
            else:
                reg.append({'shape': shape, 'pars': pars})
        if frame is not None:
            self.set('frame '+fno0)
        self.set('region system '+sys0)
        return reg

    def aperture(self, frame=None, zerobased=True):
        '''Extract apertures from circular or annulus regions in
        a list of photutils.Aperture instances'''
        reg = self.region(frame=frame, system='image', zerobased=zerobased)
        if reg == []:
            return None
        apts = []
        for r in reg:
            if isinstance(r, Region):
                if r.shape == 'circle':
                    apts.append(P.CircularAperture((r.x,r.y), r.r))
                elif r.shape == 'annulus':
                    apts.append(P.CircularAnnulus((r.x,r.y), r.r_in, r.r_out))
                else:
                    pass
            else:
                pass
        return apts

    def define_aperture(self, radius=3., centroid=False, **kwargs):
        '''Interactively define apertures

        Method takes the keywords accepted by centroid(), except for `center',
        for centroiding.
        '''
        import jylipy
        import photutils as P
        tmp = kwargs.pop('center', None)
        verbose = kwargs.pop('verbose', True)
        if not hasattr(radius, '__iter__'):
            radius = [radius]
        nr = len(radius)
        aperture = []
        i = 0
        while i<nr:
            print()
            print('Press q to exit')
            print('Left click in the image to define aperture center')
            print()
            key = self.get('imexam any coordinate image').split()
            if key[0] == 'q':
                break
            if len(key) == 3:
                x, y = key[1:]
                x, y = float(x), float(y)
                if centroid:
                    y, x = jylipy.centroid(self.get_arr2np(),center=[y,x],verbose=verbose)
                aperture.append(P.CircularAperture((x,y), radius[i%nr]))
                self.set('regions','image; circle('+','.join([str(x),str(y),str(radius[i%nr])])+')')
                i += 1
                print(('Aperture 1: ({0}, {1})'.format(x, y)))
                print()
        return aperture

    def apphot(self, **kwargs):
        from .core import apphot
        return apphot(ds9=self, **kwargs)

    def show_aperture(self, aperture, frame=None, zerobased=True):
        '''Show aperture as region'''

        if frame is not None:
            fno0 = self.get('frame')
            if str(frame) != fno0:
                self.set('frame '+str(frame))

        if not isinstance(aperture, list):
            if isinstance(aperture, P.Aperture):
                pos = aperture.positions
                if hasattr(aperture, 'r'):  # circular aperture
                    for x, y in pos:
                        if zerobased:
                            x, y = x+1, y+1
                        self.set('regions', 'image; circle('+','.join([str(x),str(y),str(aperture.r)])+')')
                elif hasattr(aperture, 'r_in'):  # annulus aperture
                    for x, y in pos:
                        if zerobased:
                            x, y = x+1, y+1
                        self.set('regions', 'image; annulus('+','.join([str(x),str(y),str(aperture.r_in),str(aperture.r_out)])+')')
                else:
                    pass
            else:
                l = len(aperture)
                if l == 3:  # circular aperture
                    x,y,r = aperture
                    self.set('regions', 'image; circle('+','.join([str(x),str(y),str(r)])+')')
                elif l == 4:  # annulus aperture
                    x,y,r1,r2 = aperture
                    self.set('regions', 'image; annulus('+','.join([str(x),str(y),str(r1),str(r2)])+')')
                else:
                    pass
        else:
            for apt in aperture:
                self.show_aperture(apt)

        if frame is not None:
            self.set('frame '+fno0)

    def sets(self, par, buf=None, blen=-1):
        '''XPA set that accepts a single command line or an array of lines'''
        if isinstance(par, str):
            return self.set(par, buf=buf, blen=blen)
        else:
            st = []
            if not hasattr(buf, '__iter__'):
                buf = [buf]*len(par)
            if not hasattr(blen, '__iter__'):
                blen = [blen]*len(par)
            for p, b, l in zip(par, buf, blen):
                st.append(self.set(p, b, l))
            return st

    def gets(self, par=None):
        '''XPA get that accepts a single command line or an array of lines'''
        if isinstance(par, str) or (par is None):
            return self.get(par)
        else:
            out = []
            for p in par:
                out.append(self.get(p))
            return out

    def backup(self, bckfile):
        '''Backup the current session'''
        self.set('backup '+bckfile)

    def restore(self, bckfile):
        '''Restore DS9 session'''
        self.set('restore '+bckfile)

    def saveimage(self, outfile, all=False):
        '''Save frame(s) to images.

        Parameters
        ----------
        outfile : str
          The full name of output file.  If multiple frames
          are to be saved, then the sequence number will be
          inserted right before the name extension, starting
          from 0.

        If `saveall` = True, then all active frames will be saved,
        with the current frame the first.

        v1.0.0 : 5/8/2015, JYL @PSI
        '''
        if len(outfile.split('.')) < 2:
            raise ValueError('The format of image file is not specified.  Please include an extension in the file name.')

        from os.path import basename
        if all:
            nfm = len(self.n_actives)
            tmp = outfile.split('.')
            fmtstr = '.'.join(tmp[:-1])+'_%0'+repr(int(np.ceil(np.log10(nfm))))+'d'+'.'+tmp[-1]
            for i in range(nfm):
                self.set('saveimage '+fmtstr % i)
                self.set('frame next')
        else:
            self.set('saveimage '+outfile)

    def setall(self, cmd, all=False):
        '''Set XPA command(s) to all active frames

        v1.0.0 : 5/8/2015, JYL @PSI
        '''
        if all:
            frm = self.frames
        else:
            frm = self.actives
        cf = self.get('frame')
        for f in frm:
            self.set('frame '+f)
            self.sets(cmd)
        self.set('frame '+cf)


def getds9(ds9=None, new=False, restore=None):
    '''Return a DS9 instance associated with a DS9 window.

    Parameters
    ----------
    ds9 : str, pyds9.DS9 instance, optional
      The ID of DS9 window.
    new : bool, optional
      If True, then a new window is openned unless `ds9` specifies
      an existing window.
    restore : str
      File name of the previously saved sessions to restore

    Returns
    -------
    A DS9 instance.

    If `ds9` is None, then either a new DS9 window is created and the
    associated DS9 instance is returned (`new`==True or no DS9 window
    is open), or the existing DS9 window that is openned the first is
    assicated with the returned DS9 instance (`new`=False).

    If `ds9` is specified, then the DS9 window with the same ID will
    be associated with the returned DS9 instance, or a new window with
    the specified ID will be opened.

    v1.0.1 : 5/8/2015, JYL @PSI
      Added keyword parameter `restore`
    '''
    if ds9 is not None:
        ds9_id = ds9
    else:
        ds9_id = None

    import pyds9
    if isinstance(ds9_id, pyds9.DS9):
        return ds9_id

    targs = pyds9.ds9_targets()
    if targs is not None:
        targs = [x.split()[0].split(':')[1] for x in targs]
    if ds9_id is None:
        if targs is None:
            return DS9(restore=restore)
        elif new:
            i = 1
            newid = 'ds9_'+str(i)
            while newid in targs:
                i += 1
                newid = 'ds9_'+str(i)
            return DS9(restore=restore, target=newid)
        else:
            return DS9(restore=restore, target=targs[0])
    else:
        return DS9(restore=restore, target=ds9_id)

