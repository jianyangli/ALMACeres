"""Beam convolution for temperature model

Adopted from Kaj William's package
"""

import time
from astropy.convolution import Gaussian2DKernel, convolve_fft
import astropy.units as u
from astropy.modeling.physical_models import BlackBody
import matplotlib.pyplot as plt

# from scipy.stats import ttest_ind, ttest_ind_from_stats, chisquare
#import os
#import sys
#import argparse

from jylipy import makenxy
from .core import ALMACeresImageMetaData, Beam
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

    def __init__(self, metadata, imsz=(512, 512), plot_extra_figures=False):
        self.metadata = metadata
        self._imsz = imsz
        self._plot_extra_figures = plot_extra_figures

    def deprojection(self, im_projected_model, meta):
        """De-project lon-lat projection to sky plane

        Parameters
        ----------
        im_projected_model : 2D array
            The lon-lat projection array
        meta : ALMACeresImageMetaData
            Meta data corresponding to the input image

        Return
        ------
        2D array: Deprojected image in sky plane
        """
        if self._plot_extra_figures:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.imshow(im_projected_model)
            # Reversed Greys colourmap for filled contours
            cpf = ax.contourf(im_projected_model, 10) # , cmap=cm.Greys_r)
            # Set the colours of the contours and labels so they're white where
            # the contour fill is dark (Z < 0) and black where it's light
            # (Z >= 0)
            colours = ['w' if level < 0 else 'k' for level in cpf.levels]
            cp = ax.contour(im_projected_model, 10, colors=colours)
            ax.clabel(cp, fontsize=10, colors=colours)
            plt.title('Projected model. Observation {}'.format( \
                    model_observation_number))
            plt.colorbar()
            plt.show()

        pxlscl = (meta.range * u.au *
            abs(meta.xscl * u.marcsec)).to('km',
                    u.dimensionless_angles()).value
        rc = np.array([482.64, 480.64, 445.57])

        # Define the viewpoint vector.
        # SOLon and SOLat are the sub-observer lon and lat.
        vpt = vector.Vector(1., meta.solon, meta.solat,
                type='geo', deg=True)

        lon, lat = vector.xy2lonlat(rc, vpt, pxlscl, imsz=self._imsz)
        w = np.isfinite(lon) & np.isfinite(lat)
        im_deprojected_model = np.zeros_like(lon)

        imsz = im_projected_model.shape
        x_lon = lon[w] / (2 * np.pi) * imsz[1]
        x_lon[x_lon > 179.4] = 0  # wrap to 0 for x_lon~180
        y_lat = (lat[w] + np.pi / 2) / np.pi * imsz[0]
        y_lat[y_lat > 90.0] = 0  # wrap for latitude???
        #return x_lon, y_lat
        im_deprojected_model[w] = \
            im_projected_model[np.round(y_lat).astype(int),
                               np.round(x_lon).astype(int)]

        if self._plot_extra_figures:
            plt.imshow(im_deprojected_model, cmap='gray')
            plt.title('Deprojected image. units: Jy/Beam')
            plt.colorbar()
            plt.show()

        return im_deprojected_model

    def projection(self, smoothed_data_gaussian, meta):
        """Project image to lon-lat projection

        Parameters
        ----------
        smoothed_data_gaussian : 2D array
            Image to be projected
        meta : ALMACeresImageMetaData
            Meta data corresponding to the input image

        Return
        ------
        2D array : Projected image in lon-lat
        """
        lat, lon = makenxy(-90, 90, 91, 0, 358, 180)
        vpt = vector.Vector(1., meta.solon, meta.solat,
                type='geo', deg=True)
        pxlscl = (meta.range * u.au * abs(meta.xscl *
                u.marcsec)).to('km', u.dimensionless_angles()).value
        rc = np.array([482.64, 480.64, 445.57])
        x, y = vector.lonlat2xy(lon, lat, rc, vpt, pa=meta.polepa,
                                center=np.array(self._imsz)//2, pxlscl=pxlscl)

        w = np.isfinite(x) & np.isfinite(y)
        imsize = smoothed_data_gaussian.shape
        x_coord = x[w]
        y_coord = y[w]
        im_convolved_model_projected = np.zeros_like(x)
        im_convolved_model_projected[w] = smoothed_data_gaussian \
                    [np.round(y_coord).astype(int),
                     np.round(x_coord).astype(int)]

        if self._plot_extra_figures:
            plt.figure()
            plt.imshow(im_convolved_model_projected)
            plt.title('Final Projected image. units: Jy/arcsec^2')
            plt.colorbar()
            plt.show()

        return im_convolved_model_projected

    def convolve_beam(self, im_deprojected_model, meta, frequency=265*u.GHz,
                  flux_input=True, beam=None):
        """Convolve image with ALMA Gaussian beam

        Parameters
        ----------
        im_deprojected_model :2D array
            Image to be convolved
        meta : ALMACeresImageMetaData
            Meta data corresponding to the input image
        frequency : u.Quantity (Hz) or number (default unit GHz), optional
            The frequency of temperature measurement.  Only used when conversion
            between temperature and flux is performed
        flux_input : bool, optional
            Specify whether the input is flux or temperature.  Temperature will
            be converted to flux using a Planck function before convolution with
            beam.  The default unit for flux is 'Jy/arcsec2'.
        beam : Beam class object, optional
            Beam to be used for convolution

        Return
        ------
        2D array : Convolved image
        """
        # parse out the beam characteristics. These are in milliarcseconds,
        # assuming at FWHM. Later we convert to std. dev. (sigma).
        # we first convert to pixels: 1 pixel = 4 milliarcseconds
        if beam is None:
            BMAJ_pixels = meta.bmaj/4.
            BMIN_pixels = meta.bmin/4.
            beam_angle_for_kernel = 90. + meta.bpa
        else:
            BMAJ_pixels, BMIN_pixels = beam.fwhm
            beam_angle_for_kernel = 90. +  \
                    getattr(beam, 'pa', 0*u.deg).to('deg').value

        # Convert images to flux
        if not flux_input:
            flux = planck_flux(im_deprojected_model, frequency)
            im_deprojected_model = flux.value
            flux_unit = flux.unit

        # Convolve image
        # make a kernel with characteristics of our beam:
        # astropy's convolution replaces NaNs with nearest neighbor
        # kernel-weighted interpolation

        # Gaussian: note that std. dev. sigma = FWHM / 2.35482 is assumed here.
        # note that the beam angle from ALMA is counterclockwise from N.,
        # whereas beam angle for astropy is CC from x-axis?

        kernel = Gaussian2DKernel(x_stddev=BMIN_pixels / 2.35482,
                                  y_stddev=BMAJ_pixels / 2.35482,
                                  theta=np.pi * beam_angle_for_kernel / 180.)
        smoothed_data_gaussian = convolve_fft(im_deprojected_model, kernel)

        if not flux_input:
            smoothed_data_gaussian = (smoothed_data_gaussian * flux_unit).to(
                    'K', u.brightness_temperature(frequency)).value

        if self._plot_extra_figures:
            plt.figure()
            plt.imshow(smoothed_data_gaussian, cmap='gray')
            # cmap='gray_r', vmin=0, vmax=10
            plt.title('Convolved deprojected image: Gaussian2d. units: Jy/Beam')
            plt.colorbar()
            plt.show()

        return smoothed_data_gaussian

    def __call__(self, data, frequency=265*u.GHz, flux_input=True, beam=None, benchmark=False):
        sz = data.shape
        data = data.reshape(-1, sz[-3], sz[-2], sz[-1])
        out = np.zeros_like(data)
        img = np.zeros((np.prod(sz[:-3]), sz[-3]) + tuple(self._imsz))
        t0 = time.time()
        avg = 0
        for i, d in enumerate(data):
            print('parameter set {} of {}'.format(i, len(data)), end='')
            if benchmark:
                dt = (time.time() - t0) / 60  # time for this case
                avg = (avg * i + dt) / (i + 1)  # average time each case
                rt = avg * (len(data) - i)  # remaining time
                print(', {:.3f} min, {:.3f} min to complete'.format(dt, rt),
                    end='\r')
                t0 = time.time()
            for j, im in enumerate(d):
                dproj = self.deprojection(im, self.metadata[j])
                conv = self.convolve_beam(dproj, self.metadata[j],
                    frequency=frequency, flux_input=flux_input, beam=beam)
                img[i, j] = conv
                out[i, j] = self.projection(conv, self.metadata[j])
        out = out.reshape(sz)
        img = img.reshape(sz[:-2] + tuple(self._imsz))
        self.convolved_model = out
        self.convolved_image = img
        return out


if __name__ == '__main__':
    pass
