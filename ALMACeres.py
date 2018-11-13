from jylipy import *
import numpy as np
from astropy import units
import json, inspect
from os.path import dirname, basename


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


def get_class_properties(obj, exclude_underscore=True):
    """Return all properties of input class instance in a dictionary
    """
    props = {}
    for p in dir(obj):
        props[p] = getattr(obj, p)
    for p in list(props.keys()):
        if inspect.ismethod(props[p]):
            props.pop(p)
    if exclude_underscore:
        for p in list(props.keys()):
            if p.startswith('_'):
                props.pop(p)
    return props


class MetaData():
    """
    Meta data class for imaging and spectroscopic data
    """
    def __init__(self, datafile=None, from_json=None, **kwargs):
        if datafile is not None:
            self.collect(datafile, **kwargs)
        elif from_json is not None:
            self.from_json(from_json)

    def __str__(self):
        return str(get_class_properties(self))

    def __repr__(self):
        return str(get_class_properties(self))

    def collect(self, datafile):
        """
        Collect meta data from data file
        """
        self.path = dirname(datafile)
        self.name = basename(datafile)

    def load(self, *args, file=None, **kwargs):
        """
        Load meta data from JSON string or file
        """
        if len(args) > 0:
            s = json.loads(*args, **kwargs)
        elif file is not None:
            with open(file, 'r') as f:
                s = json.load(f, **kwargs)
                f.close()
        for k, v in s.items():
            setattr(self, k, v)

    def dump(self, file=None, **kwargs):
        """Dump meta data to a JSON string or file
        """
        indent = kwargs.pop('indent', 4)
        properties = get_class_properties(self)
        if file is not None:
            with open(file, 'w') as f:
                json.dump(properties, f, indent=indent, **kwargs)
                f.close()
        else:
            return json.dumps(properties, indent=4, **kwargs)


class ALMACeresImageMetaData(MetaData):
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
            self.utc_mid = (Time(utc1)+(Time(utc2)-Time(utc1))/2.).isot
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
        sub = subcoord(self.utc_mid,'ceres')
        if spicekernel is not None:
            dummy = [spice.unload(k) for k in spicekernel]
        return sub


class CycleConfig():
    def __init__(self, file=None):
        if file is not None:
            self.load(file)

    def load(self, file):
        '''Load cycle configuration from JSON file'''
        f = open(file, 'r')
        s = json.load(f)
        for k,v in s.items():
            setattr(self, k, v)
        f.close()

    def dump(self, file):
        '''Dump cycle configuration to JSON file'''
        props = get_class_properties(self)
        with open(file, 'w') as f:
            json.dump(props, f, indent=4)
            f.close()


class CeresSPICE(object):
    metakernel = '/Users/jyli/work/Ceres/spice/ceres.kernels'
    pck_dawn = '/Users/jyli/work/Dawn/shape/20160108_DLR_HAMO/Ceres_DLR_HAMO_20160108.tpc'


class Cycle5(CeresSPICE):
        datadir = '/Users/jyli/Work/Ceres/ALMA/Data/Cycle5/12m_Processed/'
        imglst = [['ceres_startmodel_multiscaleG_phasecal1_1_2_3.fits',    '2017-10-15T10:50:54.2',  '2017-10-15T11:21:25.5'],
                  ['ceres_startmodel_multiscaleG_phasecal1_4_5_6.fits',    '2017-10-15T11:22:35.2',  '2017-10-15T11:51:13.4'],
                  ['ceres_startmodel_multiscaleG_phasecal1_7_8_9.fits',    '2017-10-15T12:15:11.5',  '2017-10-15T12:45:27.3'],
                  ['ceres_startmodel_multiscaleG_phasecal1_10_11_12.fits', '2017-10-15T12:46:34.8',  '2017-10-15T13:15:46.1'],
                  ['ceres_startmodel_multiscaleK_phasecal1_1_2.fits',   '2017-10-19T10:43:14.6', '2017-10-19T11:03:04.7'],
                  ['ceres_startmodel_multiscaleK_phasecal1_3_4.fits',   '2017-10-19T11:04:11.8', '2017-10-19T11:23:57.1'],
                  ['ceres_startmodel_multiscaleK_phasecal1_5_6.fits',   '2017-10-19T11:25:12.3', '2017-10-19T11:43:52.1'],
                  ['ceres_startmodel_multiscaleK_phasecal1_7_8.fits',   '2017-10-19T11:56:13.7', '2017-10-19T12:16:10.4'],
                  ['ceres_startmodel_multiscaleK_phasecal1_9_10.fits',  '2017-10-19T12:17:18.9', '2017-10-19T12:37:05.2'],
                  ['ceres_startmodel_multiscaleK_phasecal1_11_12.fits', '2017-10-19T12:38:19.5', '2017-10-19T12:57:02.1'],
                  ['ceres_startmodel2_multiscale_phasecal1_1.fits', '2017-10-26T11:06:34.1', '2017-10-26T11:15:59.8'],
                  ['ceres_startmodel2_multiscale_phasecal1_2.fits', '2017-10-26T11:17:01.1', '2017-10-26T11:26:17.7'],
                  ['ceres_startmodel2_multiscale_phasecal1_3.fits', '2017-10-26T11:27:22.4', '2017-10-26T11:36:57.2'],
                  ['ceres_startmodel2_multiscale_phasecal1_4.fits', '2017-10-26T11:38:08.4', '2017-10-26T11:48:23.7'],
                  ['ceres_startmodel2_multiscale_phasecal1_5.fits', '2017-10-26T11:49:33.3', '2017-10-26T12:06:05.4']]
        def __init__(self):
            self.imglst = np.array(self.imglst)
            self.files = [self.datadir+x[0] for x in self.imglst]


def aspect(filenames, utc1, utc2, outfile=None, spicekernel='/Users/jyli/work/Ceres/spice/ceres.kernels'):
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

    from os.path import basename
    import spiceypy as spice

    # extract information from headers
    fname = []
    bmaj = []
    bmin = []
    bpa = []
    xscl = []
    yscl = []
    for f in filenames:
        print(basename(f))
        fname.append(basename(f))
        imf,hdr = readfits(f, header=True, verbose=False)
        bmaj.append(hdr['bmaj'])
        bmin.append(hdr['bmin'])
        bpa.append(hdr['bpa'])
        xscl.append(hdr['cdelt1'])
        yscl.append(hdr['cdelt2'])

    # convert unit to mas
    xscl = abs(np.array(xscl))*3600.*1000
    yscl = abs(np.array(yscl))*3600.*1000
    bmaj = np.array(bmaj)*3600.*1000
    bmin = np.array(bmin)*3600.*1000
    obspar = Table([xscl, yscl, bmaj, bmin, bpa], names='xscl yscl BMAJ BMIN BPA'.split())
    for k in obspar.keys():
        obspar[k].format='%.2f'

    # calculate geometry
    if not is_iterable(spicekernel):
        spicekernel = [spicekernel]
    for k in spicekernel:
        spice.furnsh(k)
    midutc = (Time(utc1)+(Time(utc2)-Time(utc1))/2).isot
    sub = subcoord(midutc,'ceres')
    for k in spicekernel:
        spice.unload(k)
    sub.remove_column('Time')

    asp = table.hstack([obspar, sub])
    asp.add_column(Column(midutc,name='UTC_Mid'),index=0)
    asp.add_column(Column(fname,name='File'),index=0)

    if outfile is not None:
        asp.write(outfile, overwrite=True)

    return asp


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
    from os.path import basename
    cts = []
    fname = []
    for f in filenames:
        fname.append(basename(f))
        print(fname[-1])
        im = np.squeeze(readfits(f,verbose=False))
        cts.append(centroid(im, method=method, box=box))
    cts = np.array(cts)
    if outfile is not None:
        Table([fname, cts[:,1], cts[:,0]],names='File xc yc'.split()).write(outfile,overwrite=True)
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
    from os.path import basename
    bgs = []
    fnames = []
    for f in filenames:
        fnames.append(basename(f))
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
        Table([fnames, bgs[0], bgs[1], bgs[2], bgs[3], bgs[4]], names='File Background bg1 bg2 bg3 bg4'.split()).write(outfile, overwrite=True)
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
    from os.path import basename
    nfs = len(filenames)
    if not is_iterable(rapt):
        rapt = np.repeat(rapt,nfs)
    flux = []
    bms = []
    fname = []
    for f,c,r in zip(filenames,centers,rapt):
        fname.append(basename(f))
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

    fltbl = Table([fname,bms,flux],names='File BeamArea Flux'.split())
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
