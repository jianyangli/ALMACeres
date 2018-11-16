# Modele to account for subsurface thermal emission
#
# 5/11/2017, JYL @PSI

import numpy
from ..core import surface

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
