import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table
from astropy import units, constants


def ascii_read(*args, **kwargs):
    '''astropy.io.ascii.read wrapper

    Same API and functionalities as ascii.read
    Returns the extended Table class'''
    return Table(ascii.read(*args, **kwargs))


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


def writefits(imfile, data=None, header=None, name=None, append=False, clobber=False):
    '''IDL writefits emulator'''
    if append:
        hdu = fits.ImageHDU(data, header=header, name=name)
        hdulist = fits.open(imfile)
        hdulist.append(hdu)
        hdulist.writeto(imfile, clobber=True)
    else:
        hdu = fits.PrimaryHDU(data, header=header)
        hdu.writeto(imfile, clobber=clobber)


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


def rotm(phi, axis=2):
    '''
 Calculate a numpy matrix to transform a vector to a new coordinate
 that rotates from the old coordinate along a coordinate axis

 Parameters
 ----------
 phi : floating point
   Angle of rotation [deg]
 axis: integer 0, 1, or 2
   Axis of rotation.   0 for x-axis, 1 for y-axis, 2 for z-axis

 Returns
 -------
 numpy matrix, the transformation matrix from old coordinate to new
 coordinate.

 v1.0.0 : JYL @PSI, August 4, 2013.
 v1.0.1 : JYL @PSI, 2/19/2016
   Use np.deg2rad to perform deg to rad conversion
 v1.0.2 : JYL @PSI, 2/21/2016
   Change the default unit of angle `phi` to radiance to simplify future
     revision with astropy.units
    '''

    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    m = np.matrix([[cosphi,sinphi,0],[-sinphi,cosphi,0],[0,0,1]])
    if axis == 0:
        m = np.roll(np.roll(m,1,axis=0),1,axis=1)
    if axis == 1:
        m = np.roll(np.roll(m,-1,axis=0),-1,axis=1)
    return m


class Vector(np.ndarray):
    '''Vector object class

    Initialization
    --------------

        v = Vector(array, axis=0, type=0, coordinate=None)
        v = Vector(c1, c2, c3, axis=-1, type=0, coordinate=None)
        v = array.view(Vector)

    If initialized with one array, then the axis specified by keyword
    `axis` defines the three coordinate components of the array.  Default
    is the last axis.

    If initialized with three arrays, then they must have the same shape.

    Keyword `type` defines the coordinate type of input components (if
    string, then not type sensitive):

    Cartesian: defined by x, y, z; type in {0, 'cartesian', 'car'}
    Spherical: defined by r, phi, theta; type in {1, 'spherical', 'sph'}
    Cylindrical: defined by rho, phi, z; type in {2, 'cylindrical', 'cyl'}
    Geographic: defined by r, lon, lat; type in {3, 'geographic', 'geo'}

    Keywords
    --------
    type : str or int
      The type of coordinate definition.  Default is Cartesian.
    coordinate : CoordinateSystem class instance
      The coordinate system in which the vectors are defined.
    deg : bool
      If `True`, then angles are in degrees.  Default is `False`, and
      all angles are in radiance.

    Vector arithmatic operations
    ----------------------------

    Addition (+)
    Subtraction (-)
    Scaling (*)
    Negative (-)
    Equal (==)
    Not equal (!=)
    Dot multiplication (.dot)
    Cross multiplication (.cross)
    Normal (.norm)

    Scaling: A vector can only be scaled by an array(-like).  Each
      element in the array is a scaling factor for the corresponding
      vector element.
    Normal: Support any orders.  E.g., order 2 norm is the length of
      a vector in 3-D space; order 1 norm is the sum of abs(x_i) where
      x_i is the coordinate in axis i.

    All operations support numpy broadcast.


    v1.1.0 : JYL @PSI, 2/20/2016
      Changed the internal storage of vectors from structured array to
      general numpy array.
    v1.1.0 : JYL @PSI, 12/14/2016
      Bug fix in .__str__() and .reshape()
    '''

    _types = ({'code': ('cartesian', 'car'), 'colnames': 'x y z'.split()}, \
              {'code': ('spherical', 'sph'), 'colnames': 'r theta phi'.split()}, \
              {'code': ('cylindrical', 'cyl'), 'colnames': 'rho phi z'.split()}, \
              {'code': ('geographic', 'geo'), 'colnames': 'r lat lon'.split()})

    def __new__(cls, *var, **kwargs):

        if len(var) not in [1,3]:
            raise TypeError('{0} takes either 1 argument or 3 arguments ({1} given)'.format(cls, len(var)))

        axis = kwargs.pop('axis', -1)
        ctype = kwargs.pop('type', 0)
        deg = kwargs.pop('deg', False)

        if len(var) == 1:
            # if already Vector, return a copy
            if isinstance(var[0], Vector):
                return var[0].copy()

            base = np.asarray(var[0])

            # intialize with an array
            if base.ndim == 1:
                base = np.asarray(base)
            elif base.shape[axis] != 3:
                raise ValueError('the length of input array along axis {0} must be 3, length {1} received'.format(axis, base.shape[axis]))
            else:
                base = np.rollaxis(base, axis)
            b1, b2, b3 = base

        # initialized with three coordinate components
        elif len(var) == 3:
            b1, b2, b3 = var
            b1 = np.asarray(b1)
            b2 = np.asarray(b2)
            b3 = np.asarray(b3)
            l1 = np.shape(b1)
            l2 = np.shape(b2)
            l3 = np.shape(b3)
            if (l1 != l2) | (l1 != l3):
                raise ValueError('three arrays must have the same shape')

        # convert to (x,y,z) if needed
        typecode = cls._choose_type(ctype)
        if typecode == 1:
            if deg:
                b2 = np.deg2rad(b2)
                b3 = np.deg2rad(b3)
            b1, b2, b3 = cls.sph2xyz(b1, b2, b3)
        elif typecode == 2:
            if deg:
                b2 = np.deg2rad(b2)
            b1, b2, b3 = cls.cyl2xyz(b1, b2, b3)
        elif typecode == 3:
            if deg:
                b2 = np.deg2rad(b2)
                b3 = np.deg2rad(b3)
            b1, b2, b3 = cls.geo2xyz(b1, b2, b3)

        # generate object
        data = np.asarray([b1,b2,b3])
        data = np.rollaxis(data, 0, data.ndim)
        obj = data.view(Vector)
        obj.coordinate = kwargs.pop('coordinate', None)

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.coordinate = getattr(obj, 'coordinate', None)

    def __mul__(self, other):
        '''The multiplication operand applies scaling of vectors.  For
        dot product or cross product, use `.dot` or `.cross` instead.
        '''
        arr1 = np.rollaxis(self.view(np.ndarray),-1)
        arr2 = np.asarray(other)
        return Vector(arr1*arr2, axis=0)

    def __rmul__(self, other):
        '''To satisfy the commutative rule for scaling multiplication'''
        return self.__mul__(other)

    def __eq__(self, other):
        comp = self.view(np.ndarray)==np.asarray(other)
        if isinstance(comp, np.ndarray):
            comp = comp.all(axis=-1)
        return comp

    def __ne__(self, other):
        eq = self.__eq__(other)
        if isinstance(eq, np.ndarray):
            return ~eq
        else:
            return not eq

    def __str__(self):
        d = self.view(np.ndarray)
        d = np.rollaxis(d, -1)
        s = d.__str__().split('\n')
        if len(s) == 1:
            return s[0]
        else:
            s[0] = s[0][1:]
            s[1] = s[1][1:]
            s[2] = s[2][1:-1]
            s = '\n'.join(s)
            return s

    def __repr__(self):
        d = self.view(np.ndarray)
        d = np.rollaxis(d, -1)
        return d.__repr__().replace('array','Vector')

    def __len__(self):
        if self.shape == ():
            raise TypeError("'Vector' object with a scalar value has no len()")
        else:
            return super(Vector, self).__len__()

    @property
    def ndim(self):
        return self.view(np.ndarray).ndim-1

    @property
    def shape(self):
        return self.view(np.ndarray).shape[:-1]

    @property
    def x(self):
        '''Cartesian x in 1-D array'''
        return np.rollaxis(self.view(np.ndarray),-1,0)[0]

    @property
    def y(self):
        '''Cartesian y in 1-D array'''
        return np.rollaxis(self.view(np.ndarray),-1,0)[1]

    @property
    def z(self):
        '''Cartesian z in 1-D array'''
        return np.rollaxis(self.view(np.ndarray),-1,0)[2]

    @property
    def xyz(self):
        return np.rollaxis(self.view(np.ndarray),-1,0)

    @property
    def r(self):
        '''Spherical r in 1-D array'''
        return self.norm()

    @property
    def theta(self):
        '''Spherical theta (radiance) in 1-D array
        0 <= theta <= pi'''
        return np.arctan2(self.rho, self.z)

    @property
    def phi(self):
        '''Spherical phi (radiance) in 1-D array
        0 <= phi < 2 pi'''
        return np.arctan2(self.y, self.x) % (2*np.pi)

    @property
    def sph(self):
        '''Spherical (r, phi, theta) in 3xN array'''
        return np.asarray([self.r, self.phi, self.theta])

    @property
    def lat(self):
        '''Latitude (radiance) in 1-D array
        -pi/2 <= lat <= pi/2'''
        return np.arctan2(self.z, self.rho)

    @property
    def lon(self):
        '''longitude (radiance) in 1-D array
        0 <= lon < 2 pi'''
        return self.phi

    @property
    def geo(self):
        '''Geographic (r, lon, lat) in 3xN array'''
        return np.asarray([self.r, self.lon, self.lat])

    @property
    def rho(self):
        '''Cylindrical rho in 1-D array'''
        return np.sqrt(self.x*self.x+self.y*self.y)

    @property
    def cyl(self):
        '''Cylindrical (rho, phi, z) in 3xN array'''
        return np.asarray([self.rho, self.phi, self.z])

    def norm(self, order=2):
        '''Compute the normal of vector(s)'''
        import numbers
        if not isinstance(order, numbers.Integral):
            raise TypeError('`order` must be an integer type.')
        if order < 1:
            raise ValueError('`order` must be a positive integer.')
        return (np.abs(self.x)**order+np.abs(self.y)**order+np.abs(self.z)**order)**(1./order)

    def dot(self, other):
        '''dot product with another vector

        If `other` is not a Vector type, then it will be converted to a
        Vector type first.  The multiplication follows numpy array
        broadcast rules.
        '''
        if not isinstance(other, Vector):
            other = Vector(other)
        return (self.view(np.ndarray)*other.view(np.ndarray)).sum(axis=-1)

    def cross(self, other):
        '''cross product with other vector(s)

        If `other` is not a Vector type, then it will be converted to a
        Vector type first.  The cross product follows numpy array
        broadcast rules'''
        if not isinstance(other, Vector):
            other = Vector(other)
        x = self.y*other.z-self.z*other.y
        y = self.z*other.x-self.x*other.z
        z = self.x*other.y-self.y*other.x
        return Vector(x, y, z)

    def reshape(self, *var):
        if isinstance(var[0],tuple):
            var = (3,)+var[0]
        else:
            var = (3,)+var
        v = np.rollaxis(self.view(np.ndarray),-1)
        v = v.reshape(*var)
        return Vector(v, axis=0)

    def flatten(self):
        return self.reshape(-1)

    def vsep(self, v2, axis=-1, type=0, deg=False, directional=False):
        '''Angular separation to another vector.  Calculation follows
        numpy array broadcast rules
        '''
        if not isinstance(v2, Vector):
            v2 = Vector(v2, axis=axis, type=type)
        angle = np.arccos(self.dot(v2)/(self.norm()*v2.norm()))
        if directional:
            pi2 = np.pi*2
            zcomp = self.x*v2.y - self.y*v2.x
            wz = zcomp < 0
            if angle[wz].size > 0:
                if hasattr(angle, '__iter__'):
                    angle[wz] = pi2 - angle[wz]
                else:
                    angle = pi2 - angle
            wz = zcomp == 0
            if angle[wz].size > 0:
                xcomp = self.y*v2.z - self.z*v2.y
                wx = xcomp < 0
                if angle[wz&wx].size > 0:
                    if hasattr(angle, '__iter__'):
                        angle[wz&wx] = pi2 - angle[wx&wz]
                    else:
                        angle = pi2 - angle
                wx = xcomp == 0
                if angle[wx].size > 0:
                    ycomp = self.z*v2.x - self.x*v2.z
                    wy = ycomp < 0
                    if angle[wz&wx&wy].size > 0:
                        if hasattr(angle, '__iter__'):
                            angle[wz&wx&wy] = pi2 - angle[wz&wx&wy]
                        else:
                            angle = pi2 - angle
        if deg:
            angle = np.rad2deg(angle)
        return angle

    def rot(self, phi, axis=2, deg=True):
        '''Rotate vector(s) along an axis

        `phi` must be a scalar.  Broadcast is not supported'''
        if deg:
            phi = np.deg2rad(phi)
        return VectRot(rotm(phi, axis=axis).T)*self

    def eular(self, phi, theta, psi, deg=True):
        '''Rotate vector(s) by three Eular angles

        Angles `phi`, `theta`, and `psi` must be scalars.  Broadcast is
        not supported'''
        if deg:
            phi = np.deg2rad(phi)
            theta = np.deg2rad(theta)
            psi = np.deg2rad(psi)
        return VectRot(eularm(phi, theta, psi).T)*self

    def astable(self, type=0):
        typecode = self._choose_type(type)
        names = self._types[typecode]['colnames']
        if typecode == 0:
            c1, c2, c3 = self.xyz
        elif typecode == 1:
            c1, c2, c3 = self.sph
        elif typecode == 2:
            c1, c2, c3 = self.cyl
        elif typecode == 3:
            c1, c2, c3 = self.geo
        return Table((c1.flatten(),c2.flatten(),c3.flatten()), names=names)

    @staticmethod
    def _choose_type(ctype):
        if ctype in [0,1,2,3]:
            return ctype
        if isinstance(ctype, str):
            ctype = ctype.lower()
        idx = [ctype in x['code'] for x in Vector._types]
        if True in idx:
            return idx.index(True)
        else:
            raise ValueError('Undefined coordinate type %s' % ctype)

    @staticmethod
    def sph2xyz(r, phi, theta):
        z = r*np.cos(theta)
        rho = r*np.sin(theta)
        x = rho*np.cos(phi)
        y = rho*np.sin(phi)
        return x, y, z

    @staticmethod
    def cyl2xyz(rho, phi, z):
        x = rho*np.cos(phi)
        y = rho*np.sin(phi)
        return x, y, z

    @staticmethod
    def geo2xyz(r, lon, lat):
        z = r*np.sin(lat)
        rho = r*np.cos(lat)
        x = rho*np.cos(lon)
        y = rho*np.sin(lon)
        return x, y, z


class VectRot(np.ndarray):
    '''Vector rotation class in 3-D space, including rotation and scaling

    Initialization
    --------------

    By another VectRot class instance
    By rotation matricies, and scaling factors

    To apply a rotation to a Vector:
        r = VectRot(m)
        v1 = r.inner(v)
    or equivalently
        v1 = r(v)
    or equivalently
        v1 = r*v
    Both r and v can be array of rotations and vectors, respectively.

    Operands
    --------

    Multiplication (*, .inner, or function call by class instance):
      Scale VectRot, rotate a Vector, or combine rotation with another
      VectRot.  Note that __rmul__ is defined as the same as __mul__ but
      with the transposed rotation.
    Power (**): Only defined with the power index is a scalor integer type.
    All other operands are blocked (+, -, /)

    All operations supports broadcast.


    v1.0.0 : JYL @PSI, 2/21/2016
    '''

    def __new__(cls, *var, **kwargs):

        if len(var) != 1:
            raise TypeError('{0} takes 1 argument ({1} given)'.format(cls, len(var)))

        scale = kwargs.pop('scale', None)

        # if a VectRot instance, return a copy
        if isinstance(var[0], VectRot):
            return var[0].copy()

        # initialize with numpy array
        data = np.asarray(var[0])
        if data.shape[-2:] != (3,3):
            raise ValueError('the shape of the last two dimentions of input array must be 3, length {0} received'.format(data.shape[-2:]))

        # attach scaling factors
        if scale is not None:
            scale = np.asarray(scale)
            s = np.stack([scale]*3,axis=-1)
            s = np.stack([s]*3,axis=-1)
            data = data*s

        # generate object
        return data.view(VectRot)

    def __str__(self):
        d = self.view(np.ndarray)
        d = np.rollaxis(d, -1)
        d = np.rollaxis(d, -1)
        s = d.__str__()
        s = [x[1:] for x in s.split('\n')]
        s[2] = s[2][:-1]
        s = '\n'.join(s)
        return s

    def __repr__(self):
        d = self.view(np.ndarray)
        d = np.rollaxis(d, -1)
        d = np.rollaxis(d, -1)
        return d.__repr__().replace('array', 'VectRot')

    def __len__(self):
        if self.shape == ():
            raise TypeError("'VectRot' object with a scalar value has no len()")
        else:
            return super(VectRot, self).__len__()

    def __add__(self, v):
        raise TypeError('add is not defined for VectRot type')

    def __radd__(self, v):
        self.__add__(v)

    def __sub__(self, v):
        raise TypeError('sub is not defined for VectRot type')

    def __rsub__(self, v):
        self.__sub__(v)

    def __mul__(self, v):
        return self.inner(v)

    def __rmul__(self, v):
        return self.T.inner(v)

    def __div__(self, v):
        raise TypeError('div is not defined for VectRot type')

    def __rdiv__(self, v):
        self.__div__(v)

    def __pow__(self, v):
        if hasattr(v, '__iter__'):
            raise TypeError('power with non-scaler not defined for VectRot type')
        import numbers
        if not isinstance(v, numbers.Integral):
            raise TypeError('power can only be performed with integer types')
        out = self.copy()
        while v>1:
            out = out.inner(out)
            v -= 1
        return out

    @property
    def ndim(self):
        return self.view(np.ndarray).ndim-2

    @property
    def shape(self):
        return self.view(np.ndarray).shape[:-2]

    @property
    def det(self):
        from numpy.linalg import det
        return det(self)

    @property
    def T(self):
        return VectRot(np.rollaxis(self.view(np.ndarray),-1,-2))

    def inner(self, v):
        '''Inner product of the VectRot type with another variable.

        Type `v`        Operation           Return
        -----------     ----------------    -------
        numpy array     scaling             VectRot
        Vector          Vector rotation     Vector
        VectRot         Combined VectRot    VectRot

        Support numpy broadcast rules.'''

        if isinstance(v, Vector):
            d = np.stack([v.view(np.ndarray)]*3,axis=-2)
            y = (self.view(np.ndarray)*d).sum(axis=-1)
            return Vector(y)
        elif isinstance(v, VectRot):
            d = np.rollaxis(v.view(np.ndarray),-1,-2)
            d = np.stack([d]*3, axis=-3)
            s = np.stack([self.view(np.ndarray)]*3, axis=-2)
            y = (s*d).sum(axis=-1)
            return VectRot(y)
        else:
            v = np.asarray(v)
            d = np.stack([v]*3, axis=-1)
            d = np.stack([d]*3, axis=-1)
            y = VectRot(self.view(np.ndarray)*d)
            return y

    def __call__(self, v):
        return self.inner(v)


def vect2proj(vect, viewpt, pa=0.):
    '''Convert vector coordinates to its coordinates in a parallel
    projected plane defined by `viewpt`.

    The projection plane is defined as +x along horizontal direction
    towards right, +y along verticle direction towards up.  The +z
    axis completes the right-hand system towards viewer.

    Parameter
    ---------
    vect : Vector
      The vector to be projected
    viewpt : Vector
      The vector of the view point
    pa : number, optional
      Position angle of the z-axis in projection plane, measured in
      radiance from up to left (ccw).

    Return
    ------
    A Vector instance containing the projected coordinates of the input
    Vector.

    v1.0.0 : JYL @PSI, 2/21/2016
    '''
    # Rotate along z-axis by sub-observer longitude
    m1 = VectRot(rotm(-viewpt.lon-np.pi/2, axis=2).T)
    # Rotate along x-axis by sub-observer azimuth
    m2 = VectRot(rotm(-viewpt.theta, axis=0).T)
    # Rotate along z-axis by position angle
    m3 = VectRot(rotm(pa, axis=2).T)
    return m3*m2*m1*vect


def lonlat2xy(lon, lat, r, viewpt, pa=0., center=None, pxlscl=None, deg=True):
    '''Convert the body-fixed (lon, lat) coordinates to the corresponding
    (x,y) pixel position in a CCD

    lon, lat : array-like
      The longitude and latitude to be converted.  They must have the
      same shape.
    r : number of sequence of numbers
      If a scalar or a 1-element sequence, it's the radius of a sphere.
      If a 2- or 3-element sequence, then it defines the (a, c) or
        (a, b, c) of an ellipsoid.
    viewpt : Vector
      The view point vector in body-fixed frame of the object.  Only
      parallel projection is considered, so the distance of viewer
      `viewpt.norm()` doesn't matter.
    pa : scalar, optional
      Position angle of the z-axis in image plane, measured from up to
      left (ccw).
    center : 2-element sequence, optional
      The CCD coordinates (y0,x0) of body center.  If `None`, then the
      body center will be at (0,0) image coordinates
    pxlscl : scalar number, optional
      The size of pixel in the same unit as `body`.  If `None`, then
      the pixel size will be assumed to be 1 with the same unit as
      `r`.
    deg : bool, optional
      The unit of input `lon` and `lat`.

    Return
    ------
    x, y : two arrays
      Two arrays of the same shape as `lon` and `lat` containing the
      corresponding (x, y) image coordinates.

    Algorithm
    ---------
    Calculate the (x,y,z) for input (lon,lat) using the shape model
    defined by `r`, discard those with surface normal pi/2 away from
    `viewpt`, convert to image plane, return (x,y)

    v1.0.0 : JYL @PSI, 2/24/2016
    '''
    lon = np.asarray(lon).astype(float)
    lat = np.asarray(lat).astype(float)
    if lon.shape != lat.shape:
        raise ValueError('`lon` and `lat` must have the same shape, {0} {1} received'.format(lon.shape, lat.shape))

    # set up shape
    if hasattr(r, '__iter__'):
        if len(r) == 1:
            a = r[0]
            b = a
            c = a
        elif len(r) == 2:
            a, c = r
            b = a
        elif len(r) == 3:
            a, b, c = r
        else:
            raise Warning('`r` has {0} elements, the first three elements define the triaxial ellpsoid shape, others are discarded')
            a, b, c = r[:3]
    else:
        a = r
        b = a
        c = a

    # calculate projection
    if deg:
        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)
        pa = np.deg2rad(pa)
    coslon = np.cos(lon)
    sinlon = np.sin(lon)
    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    v = Vector(a*coslon*coslat, b*sinlon*coslat, c*sinlat)
    angle = Vector(v.x/a**2, v.y/b**2, v.z/c**2).vsep(viewpt)
    w = angle<(np.pi/2)  # only keep where normal is within pi/2 of viewpt
    v = vect2proj(v[w], viewpt, pa=pa)
    x = np.ones_like(lon)*np.nan
    y = np.ones_like(lon)*np.nan
    x[w] = v.x
    y[w] = v.y

    # add pixel scale and pixel center
    if pxlscl is not None:
        x /= pxlscl
        y /= pxlscl
    if center is not None:
        x += center[1]
        y += center[0]

    return x, y


class Ceres(object):
    # body shape and pole based on DLR RC3 shape model
    ra = 482.64 * units.km
    rb = 480.60 * units.km
    rc = 445.57 * units.km
    r = np.sqrt((ra+rb)*rc/2)
    pole = (291.41, 66.79)
    GM = 62.68 * units.Unit('km3/s2')
    M = (GM/constants.G).decompose()


if __name__ == '__main__':

    datadir = '/Users/jyli/work/Ceres/ALMA/Data/Cycle3/Processed/CERES_c2/'
    aspfile = '/Users/jyli/work/Ceres/ALMA/Data/Cycle3/Processed/Ceres_c.csv'
    ctfile = '/Users/jyli/work/Ceres/ALMA/Modeling/center_c.csv'

    asp = ascii_read(aspfile)
    ct = ascii_read(ctfile)

    ims = []
    lsts = []
    emis = []
    rc = Ceres.ra.value, Ceres.rb.value, Ceres.rc.value
    for r,c in zip(asp,ct):
        # load image
        fname = datadir+r['File'].replace('_selfcal','')
        print(fname)
        im = readfits(fname,verbose=False)
        im = np.squeeze(im)
        im /= np.pi*r['BMAJ']*r['BMIN']*1e-6/4

        # project to (lon, lat)
        lat,lon = makenxy(-90,90,91,0,358,180)
        vpt = Vector(1., r['SOLon'], r['SOLat'], type='geo', deg=True)
        pxlscl = r['Range']*1.496e8*r['xscl']/206265000.
        x,y = lonlat2xy(lon,lat,rc,vpt,pa=r['PolePA'],center=(c['yc'],c['xc']),pxlscl=pxlscl)
        w = np.isfinite(x) & np.isfinite(y)
        b = np.zeros_like(x)
        b[w] = im[np.round(y[w]).astype(int),np.round(x[w]).astype(int)]

        # calculate local solar time
        lst = ((lon-r['SSLon'])/15+12) % 24

        # calculate emission angle
        emi = Vector(np.ones_like(lon),lon,lat,type='geo',deg=True).vsep(Vector(1,r['SOLon'],r['SOLat'],type='geo',deg=True))

        # save projection
        writefits('/Users/jyli/work/Ceres/ALMA/Modeling/projected/'+r['File'],b,  clobber=True)
        writefits('/Users/jyli/work/Ceres/ALMA/Modeling/projected/'+r['File'],lst ,name='LST',append=True)

        # collect images and lst
        ims.append(b)
        lsts.append(lst)
        emis.append(emi)

    ims = np.array(ims)
    lsts = np.array(lsts)
    emis = np.array(emis)
    data = np.stack((ims,lsts,emis))
    hdr = fits.Header()
    hdr['bunit'] = 'Jy/arcsec**2'
    writefits('/Users/jyli/work/Ceres/ALMA/Modeling/fluxes_lst_c.fits',data,hdr,clobber=True)
