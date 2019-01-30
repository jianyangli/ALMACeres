# Test layered subsurface emission model
#
# 1/29/2019, JYL @PSI

import numpy
from ALMACeres.core import Surface, Layer, loss_tangent
import copy

from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from jylipy import pplot
import os

workdir = os.path.dirname(__file__)

pdf = PdfPages(os.path.join(workdir,'layered_subsurface_model_test.pdf'))

# generate layers
TT = [lambda z: numpy.clip(300-z,50,None),
      lambda z: numpy.clip(200-z,50,None),
      lambda z: numpy.clip(100-z,50,None),
      lambda z: numpy.clip(80-z,50,None)]
dd = [100,100,20,numpy.inf]
ns = [1.3, 1.5, 1.7, 1.9]
loss = 0.0067
layers = []
for n,d,T in zip(ns,dd,TT):
    layers.append(Layer(n, loss, depth=d, profile=T))
#for l in layers:
#    print(l.profile(linspace(0,100,5)))

zz = []
tt = []
L0 = 0
for l in layers:
    if l.depth != numpy.inf:
        z = numpy.linspace(0,l.depth,100)
    else:
        z = numpy.linspace(0,1000,100)
    tt.append(l.profile(z))
    zz.append(z+L0)
    L0 += l.depth
zz = numpy.concatenate(zz)
tt = numpy.concatenate(tt)

plt.clf()
plt.plot(zz,tt)
plt.vlines(numpy.array([0]+[l.depth for l in layers]).cumsum(),tt.min(),tt.max(),linestyle='--')
pplot(xlabel='z-depth',ylabel='T (K)',title='Assumed Temperature Profile')
pdf.savefig()

# brightness temperature as a function of emissiona angle
surf1 = Surface(layers)  # layered model
surf2 = Surface(copy.deepcopy(layers[0]))  # single layer model
surf2.layers[0].depth = numpy.inf

wavelength = 10.   # wavelength
emi = numpy.linspace(0,90,100)
Tb1 = numpy.array([surf1.emission(ee, wavelength) for ee in emi])
Tb2 = numpy.array([surf2.emission(ee, wavelength) for ee in emi])
plt.clf()
plt.plot(emi, Tb1)
plt.plot(emi, Tb2)
pplot(xlabel='Emission Angle (deg)',ylabel='Tb (K)')
plt.legend(['4 layer model','single layer model'])
pdf.savefig()

pdf.close()
