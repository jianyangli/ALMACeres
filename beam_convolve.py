"""Beam convolution for temperature model

Adopted from Kaj William's package
"""

import datetime
from astropy import *
import matplotlib.pyplot as plt
import time
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel
from astropy.convolution import convolve, convolve_fft
from scipy.stats import ttest_ind, ttest_ind_from_stats, chisquare
from astropy.io import fits
import astropy.units as u
from astropy.modeling.physical_models import BlackBody
import os
import sys
import argparse

#from jylipy import
from .core import ALMACeresImageMetaData, Beam
from .utils import *
from . import vector


def planck_flux(T, freq):
    """Calculate thermal flux using Planck function

    Parameters
    ----------
    T : number, Quantity, or array of them
        Temperature in K by default
    freq : number, Quantity, or array of them
        Frequency for calculation, in GHz by default.  If both T and
        freq are arrays, then numpy broadcast rules apply.

    Return
    ------
    Quantity, thermal flux
    """
    T = u.Quantity(T, 'K')
    freq = u.Quantity(freq, 'GHz')
    return BlackBody(T)(freq)


class BeamConvolve:

    def __init__(self):
        self.metadata = metadata
        self.imsz = imsz
        self.plot_extra_figures = plot_extra_figures

    def deprojection(self, im_projected_model, metadata, imsz=(512, 512), plot_extra_figures=False):
        pass

    def projection(self, smoothed_data_gaussian, metadata, plot_extra_figures=False):
        pass

    def convolve_beam(im_deprojected_model, metadata, frequency=265*u.GHz,
                  flux_input=True, beam=None,
                  plot_extra_figures=False):
        pass

    def __call__():
        pass


if __name__ == '__main__':
    pass
