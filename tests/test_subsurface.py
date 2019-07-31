# Modele to account for subsurface thermal emission
#
# 5/11/2017, JYL @PSI

import numpy
from ALMACeres.core import Surface, Layer, loss_tangent

# Some tests
if __name__ == '__main__':
    from scipy.interpolate import interp1d
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    #from jylipy import pplot
    import os

    workdir = os.path.dirname(__file__)

    pdf = PdfPages(os.path.join(workdir,'subsurface_emission_test.pdf'))

    # temperature profile
    zz = numpy.linspace(0,300,1000)
    TT = numpy.linspace(300,100,1000)
    T = interp1d(zz, TT, bounds_error=False, fill_value=100)
    plt.clf()
    z = numpy.linspace(0,500,1000)
    plt.plot(z,T(z))
    #pplot(xlabel='z-depth',ylabel='T * Emissivity')
    #pdf.savefig()

    # optical constants
    n = numpy.sqrt(3)  # refractive index

    # brightness temperature as a function of emissiona angle
    loss = 0.0067   # loss tangent
    wavelength = 10.   # wavelength
    layer = Layer(n, loss, profile=T)
    surf = Surface(layer)
    emi = numpy.linspace(0,90,100)
    Tb = numpy.array([surf.emission(ee, wavelength) for ee in emi])
    plt.clf()
    plt.plot(emi, Tb)
    #pplot(xlabel='Emission Angle (deg)',ylabel='Tb')
    pdf.savefig()

    # brightness temperature as a function of loss tangent
    loss = numpy.logspace(-5,-1,100)  # loss tangent
    wavelength = 10.   # wavelength
    Tb = []
    leng = []
    for ls in loss:
        surf.layers[0].loss_tangent = ls
        Tb.append(surf.emission(0, wavelength))
        leng.append(surf.layers[0].absorption_length(wavelength))
    plt.clf()
    plt.plot(loss, Tb)
    #pplot(xscl='log',xlabel='Loss Tangent',ylabel='Tb')
    pdf.savefig()

    # brightness temperature as a function of wavelength
    loss = 0.0067  # loss tangent
    wavelength = numpy.logspace(-2,3.5,100)  # wavelength
    layer = Layer(n, loss,  profile=T)
    surf = Surface(layer)
    Tb = [surf.emission(0, wavelength=x) for x in wavelength]
    plt.clf()
    plt.plot(wavelength, Tb)
    #pplot(xscl='log',xlabel='Wavelength',ylabel='Tb')
    pdf.savefig()

    pdf.close()


    # test the effects of refractive index and absorption length on effective
    # brightness temperature
    pdf = PdfPages(os.path.join(workdir,'test_Le_n.pdf'))

    # assumed temperature profile
    zz = numpy.linspace(0,300,1000)
    TT = numpy.linspace(300,100,1000)
    T = interp1d(zz, TT, bounds_error=False, fill_value=100)
    plt.clf()
    z = numpy.linspace(0,500,1000)
    plt.plot(z,T(z))
    #pplot(xlabel='z-depth',ylabel='T')
    pdf.savefig()

    # wavelength is 1/10 of temperature profile
    wavelength = 10.

    # test effect of absorption length Le
    n = 1.5
    wavelength = 10.
    Le = numpy.logspace(-1,4,100)
    Tb = []
    loss_s = []
    layer = Layer(n, loss, profile=T)
    surf = Surface(layer)
    for l in Le:
        loss = loss_tangent(n, l, wavelength)
        loss_s.append(loss)
        surf.layers[0].loss_tangent = loss
        Tb.append(surf.emission(0., wavelength))
    plt.clf()
    plt.plot(Le, Tb)
    #pplot(xlabel='Absorption Length (wavelength)',ylabel='Tb (K, for T$_0$=300 K)',xscl='log',title='n={0}'.format(n))
    pdf.savefig()

    # test effect of refractive index
    loss = 0.0067   # loss tangent
    wavelength = 10.   # wavelength
    rn = numpy.linspace(1.,3,5)
    emi = numpy.linspace(0,89.9,30)
    Tb = []
    layer = Layer(n, loss, profile=T)
    surf = Surface(layer)
    for n in rn:
        surf.layers[0].n = n
        Tb.append([surf.emission(ee, wavelength) for ee in emi])
    Tb = numpy.array(Tb).T
    plt.clf()
    plt.plot(emi, Tb)
    #pplot(xlabel='Emission Angle (deg)', ylabel='Tb (K)', title=r'$\tan(\Delta)=0.0067$, $\lambda=10$')
    plt.legend([f'n = {i:.2f}' for i in rn])
    pdf.savefig()

    pdf.close()
