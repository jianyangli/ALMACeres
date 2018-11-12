# Modele to account for subsurface thermal emission
#
# 5/11/2017, JYL @PSI

import numpy

class snell(object):

    def __init__(self, n1, n2=1., deg=True):
        self.n1 = n1
        self.n2 = n2
        self.deg = True

    def angle1(self, angle2):
        if self.deg:
            angle2 = numpy.deg2rad(angle2)
        a1 = numpy.arcsin(self.n2/self.n1 * numpy.sin(angle2))
        if self.deg:
            a1 = numpy.rad2deg(a1)
        return a1

    def angle2(self, angle1):
        if self.deg:
            angle1 = numpy.deg2rad(angle1)
        a2 = numpy.arcsin(self.n1/self.n2*numpy.sin(angle1))
        if self.deg:
            a2 = numpy.rad2deg(a2)
        return a2


# absorption length
absorption_length = lambda n, loss_tangent, wavelength=1.: wavelength/(4*numpy.pi*n)*(2./((1+loss_tangent*loss_tangent)**0.5-1))**0.5

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
            self.loss_tangent = ((2*(wavelength/(4*numpy.pi*self.n*absorption_length))**2+1)**2-1)**0.5
        if self.loss_tangent is None:
            raise ValueError('either `loss_tangent` or `absorption_length` has to be specified')

    def Tb(self, emi, wavelength, epsrel=1e-4):
        '''Calculate brightness temperature with subsurface emission
        accounted for'''
        if hasattr(emi,'__iter__'):
            emi = numpy.asanyarray(emi)
            results = numpy.zeros_like(emi).flatten()
            emi_flat = emi.flatten()
            for i in range(len(results)):
                results[i] = self.Tb(emi_flat[i], wavelength=wavelength, epsrel=epsrel)
            return results.reshape(emi.shape)

        s = snell(n)
        inc = s.angle1(emi)
        coef = 1/self.absorption_length(wavelength)   # absorption coefficient
        sec_i = 1./numpy.cos(numpy.deg2rad(inc))
        intfunc = lambda z: self.T(z) * numpy.exp(-coef*sec_i*z)
        from scipy.integrate import quad
        integral = quad(intfunc, 0, numpy.inf, epsrel=epsrel)[0]
        return self.emissivity*sec_i*coef * integral

    def absorption_length(self, wavelength=1.):
        '''Electrical skin depth, or absorption length
        If wavelength is not specified, then it returns Le in unit of wavelength'''
        return absorption_length(self.n, self.loss_tangent, wavelength)


# Some tests
if __name__ == '__main__':
    from scipy.interpolate import interp1d
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from jylipy import pplot

    pdf = PdfPages('subsurface_emission_test.pdf')

    # temperature profile
    zz = numpy.linspace(0,300,1000)
    TT = numpy.linspace(300,100,1000)
    T = interp1d(zz, TT, bounds_error=False, fill_value=100)
    plt.clf()
    z = numpy.linspace(0,500,1000)
    plt.plot(z,T(z))
    pplot(xlabel='z-depth',ylabel='T * Emissivity')
    pdf.savefig()

    # optical constants
    n = numpy.sqrt(3)  # refractive index

    # brightness temperature as a function of emissiona angle
    loss = 0.0067   # loss tangent
    wavelength = 10.   # wavelength
    surf = surface(T, n, loss, emissivity=1.0)
    emi = numpy.linspace(0,90,100)
    Tb = surf.Tb(emi, wavelength)
    plt.clf()
    plt.plot(emi, Tb)
    pplot(xlabel='Emission Angle (deg)',ylabel='Tb')
    pdf.savefig()

    # brightness temperature as a function of loss tangent
    loss = numpy.logspace(-5,-1,100)  # loss tangent
    wavelength = 10.   # wavelength
    Tb = []
    leng = []
    for ls in loss:
        surf = surface(T, n, ls, emissivity=1.0)
        Tb.append(surf.Tb(0, wavelength))
        leng.append(surf.absorption_length())
    plt.clf()
    plt.plot(loss, Tb)
    pplot(xscl='log',xlabel='Loss Tangent',ylabel='Tb')
    pdf.savefig()

    # brightness temperature as a function of wavelength
    loss = 0.0067  # loss tangent
    wavelength = numpy.logspace(-2,3.5,100)  # wavelength
    surf = surface(T, n, loss, emissivity=1.0)
    Tb = [surf.Tb(0, wavelength=x) for x in wavelength]
    plt.clf()
    plt.plot(wavelength, Tb)
    pplot(xscl='log',xlabel='Wavelength',ylabel='Tb')
    pdf.savefig()

    pdf.close()


    # test the effects of refractive index and absorption length on effective
    # brightness temperature
    pdf = PdfPages('test_Le_n.pdf')

    # assumed temperature profile
    zz = numpy.linspace(0,300,1000)
    TT = numpy.linspace(300,100,1000)
    T = interp1d(zz, TT, bounds_error=False, fill_value=100)
    plt.clf()
    z = numpy.linspace(0,500,1000)
    plt.plot(z,T(z))
    pplot(xlabel='z-depth',ylabel='T')
    pdf.savefig()

    # wavelength is 1/10 of temperature profile
    wavelength = 10.

    # test effect of absorption length Le
    n = 1.5
    Le = numpy.logspace(-1,4,100)
    Tb = []
    for l in Le:
        surf = surface(T, n, 0., absorption_length=l)
        Tb.append(surf.Tb(0., wavelength))
    plt.clf()
    plt.plot(Le/wavelength, Tb)
    pplot(xlabel='Absorption Length (wavelength)',ylabel='Tb (K, for T$_0$=300 K)',xscl='log',title='n={0}'.format(n))
    pdf.savefig()

    # test effect of refractive index
    Le = 10
    rn = numpy.linspace(1.,3,20)
    emi = numpy.linspace(0,89.9,20)
    Tb = []
    for n in rn:
        surf = surface(T, n, 0., absorption_length=Le)
        Tb.append(surf.Tb(emi, wavelength))
    Tb = numpy.array(Tb).T
    plt.clf()
    plt.plot(emi, Tb)
    pplot(xlabel='Emission Angle (deg)', ylabel='Tb', title='n=1 to 3, Le/wavelength=1')
    pdf.savefig()

    pdf.close()
