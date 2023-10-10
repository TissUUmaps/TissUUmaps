#!/usr/bin/env python

"""
.. module:: generate_sdf
   :platform: Linux, Windows
   :synopsis: Script for converting SVG marker atlas to signed distance field (SDF)

.. moduleauthor:: Fredrik Nysjo
"""

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import cairosvg


settings = {
    "output_res": 1024,
    "hq_res": 8192,
    "shape_id": 4,
    "atlas_size": 4,
    "scalar_factor": 8.0,
    "crop_to_shape_tile": False,
    "show_comparison": False,
}


def sdf_edt(image, threshold=0.5):
    """Returns SDF computed by euclidean distance transform (EDT) applied to
    binary thresholding of grayscale image
    """
    assert image.dtype == np.float32

    dt_inside = sp.ndimage.distance_transform_edt(image > threshold).astype(np.float32)
    dt_outside = sp.ndimage.distance_transform_edt(image <= threshold).astype(
        np.float32
    )
    sdf = np.maximum(0.0, dt_inside - 0.5) - np.maximum(0.0, dt_outside - 0.5)
    return sdf


def sdf_edt_downscaled(image, factor=4.0):
    """Returns SDF computed by EDT and then downsampling"""
    assert image.dtype == np.float32

    sdf = sdf_edt(image)
    sdf_downscaled = sp.ndimage.zoom(sdf / factor, zoom=1.0 / factor, order=1)
    return sdf_downscaled


def sdf_edt_smoothed(image, sigma=0.83):
    """Returns SDF computed by EDT and then applying Gaussian smoothing"""
    assert image.dtype == np.float32

    sdf = sdf_edt(image)
    sdf_gs = sp.ndimage.gaussian_filter(sdf, sigma=sigma, mode="nearest")
    return sdf_gs


def sdf_edt_upscaled(image, factor=4.0):
    """Returns SDF computed by upscaling the input grayscale image before
    commputing the EDT and then downsampling again
    """
    assert image.dtype == np.float32

    image_upscaled = sp.ndimage.zoom(image, zoom=factor, order=1)
    sdf = sdf_edt(image_upscaled)
    sdf_downscaled = sp.ndimage.zoom(sdf / factor, zoom=1.0 / factor, order=1)
    return sdf_downscaled


def sdf_edt_iterative(image, samples):
    """Returns SDF computed by iterative EDT, which is the EDT applied to
    multiple shifted thresholded versions of the input grayscale image
    """
    assert image.dtype == np.float32
    assert len(samples) > 0

    sdf_accum = np.zeros(image.shape, dtype=np.float32)
    for shift in samples:
        image_shifted = sp.ndimage.shift(image, shift, order=1)
        assert image_shifted.shape == image.shape
        sdf_accum += sdf_edt(image_shifted)
    return sdf_accum / len(samples)


def sdf_edt_iterative_grid(image, gridsize=4):
    """Returns SDF computed by iterative EDT with samples from uniform grid"""
    assert gridsize > 0

    samples = []
    num_samples = gridsize * gridsize
    for n in range(0, num_samples):
        shift_x = (((n + 0.5) / gridsize) % 1.0) - 0.5
        shift_y = (((n // gridsize) + 0.5) / gridsize) - 0.5
        samples.append((shift_x, shift_y))
    return sdf_edt_iterative(image, samples)


def sdf_edt_iterative_lds(image, num_samples=8):
    """Returns SDF computed by iterative EDT with samples from a low
    discrepancy sequence
    """
    assert num_samples > 0

    samples = []
    for n in range(0, num_samples):
        shift_x = ((2.0**0.5 * (n + 1)) % 1.0) - 0.5
        shift_y = ((3.0**0.5 * (n + 1)) % 1.0) - 0.5
        samples.append((shift_x, shift_y))
    return sdf_edt_iterative(image, samples)


def sdf_edt_averaging(image, num_thresholds=8, lower=0.0, upper=1.0):
    """Returns SDF computed by splitting the range [lower, upper] into multiple
    thresholds and computing the EDT for each of them and taking the average.

    This is the same method as the one implemented in the GEGL library,
    https://gitlab.gnome.org/GNOME/gegl/-/blob/master/operations/common-cxx/distance-transform.cc
    """
    assert num_thresholds > 0

    sdf_accum = np.zeros(image.shape, dtype=np.float32)
    for i in range(0, num_thresholds):
        # threshold_i = lower + (upper - lower) * ((i + 1.0) / (num_thresholds + 1.0))
        # Improved threshold value that seems to produce nicer zero-level contour
        threshold_i = lower + (upper - lower) * ((i + 0.5) / (num_thresholds))
        sdf_accum += sdf_edt(image, threshold_i)
    return sdf_accum / num_thresholds


def sdf_edt_upscaled_averaging(
    image, factor=2.0, num_thresholds=4, lower=0.0, upper=1.0
):
    """Returns SDF computed by upscaling the input grayscale image before
    computing the SDF with sdf_edt_averaging() and then downsampling again
    """
    image_upscaled = sp.ndimage.zoom(image, zoom=factor, order=1)
    sdf = sdf_edt_averaging(
        image_upscaled, num_thresholds=num_thresholds, lower=lower, upper=upper
    )
    sdf_downscaled = sp.ndimage.zoom(sdf / factor, zoom=1.0 / factor, order=1)
    return sdf_downscaled


def plot_sdf_contours(sdf, ref=None):
    """Plots SDF contours, plus optional extra contour from grayscale image"""
    plt.imshow(sdf > 0.0, cmap="gray")
    # plt.contourf(sdf, cmap="viridis", levels=[-6,-4,-2,0,2,4,6])
    plt.contourf(sdf, levels=[-3, -2, -1, 0, 1, 2, 3])
    # plt.contour(sdf, levels=[0.0])
    if ref is not None:
        # plt.contour(ref, levels=[0.5], linewidths=[0.5])
        plt.contour(sdf, levels=[0.0], linewidths=[0.5])
    plt.xticks([]), plt.yticks([])


def plot_sdf_error(sdf, ref, show_colorbar=True):
    """Plots SDF absolute error against another reference SDF image"""
    plt.imshow(abs(sdf - ref), cmap="turbo", vmin=0.0, vmax=1.0)
    plt.xticks([]), plt.yticks([])
    if show_colorbar:
        # Use "magic" fraction parameter to fit colorbar to plot
        plt.colorbar(ax=plt.gca(), fraction=0.046, pad=0.035)


def main():
    output_res = settings["output_res"]
    hq_res = settings["hq_res"]
    shape_id = settings["shape_id"]
    atlas_size = settings["atlas_size"]
    scalar_factor = settings["scalar_factor"]
    crop_to_shape_tile = settings["crop_to_shape_tile"]

    print("Loading and rasterising marker atlas from markershapes.svg...")
    cairosvg.svg2png(
        url="markershapes.svg",
        output_width=output_res,
        output_height=output_res,
        write_to=".tmp.png",
    )
    image_atlas = plt.imread(".tmp.png")[:, :, 3]
    cairosvg.svg2png(
        url="markershapes.svg",
        output_width=hq_res,
        output_height=hq_res,
        write_to=".tmp.png",
    )
    image_atlas_hq = plt.imread(".tmp.png")[:, :, 3]

    image = image_atlas
    if crop_to_shape_tile:
        tile_size = output_res // atlas_size
        x = (shape_id % atlas_size) * tile_size
        y = (shape_id // atlas_size) * tile_size
        image = image_atlas[y : y + tile_size, x : x + tile_size]

    image_hq = image_atlas_hq
    if crop_to_shape_tile:
        tile_size = hq_res // atlas_size
        x = (shape_id % atlas_size) * tile_size
        y = (shape_id // atlas_size) * tile_size
        image_hq = image_atlas_hq[y : y + tile_size, x : x + tile_size]

    print("Computing SDF from rasterised marker atlas...")
    sdf = sdf_edt_averaging(image, num_thresholds=8)

    print("Saving truncated SDF to markershapes.png")
    sdf_truncated = np.maximum(
        0.0, np.minimum(1.0, sdf * (scalar_factor / 255.0) + 0.5)
    )
    # Applying a bit of smoothing to the SDF before quantization to 8-bit
    # format makes the resulting contours look nicer
    sdf_truncated = sp.ndimage.gaussian_filter(sdf_truncated, sigma=0.5, mode="nearest")
    plt.imsave("markershapes.png", sdf_truncated, cmap="gray", vmin=0.0, vmax=1.0)

    if settings["show_comparison"]:
        print("Computing additional SDFs for method comparison...")

        sdf_hq = sdf_edt_downscaled(image_hq, factor=(hq_res / output_res))

        sdfs = []
        sdfs.append(sdf_hq)
        sdfs.append(sdf_edt(image))
        sdfs.append(sdf_edt_smoothed(image, sigma=0.83))
        sdfs.append(sdf_edt_upscaled(image, factor=4.0))
        sdfs.append(sdf_edt_iterative_grid(image, gridsize=4))
        sdfs.append(sdf_edt_iterative_lds(image, num_samples=8))
        sdfs.append(sdf_edt_averaging(image, num_thresholds=8))
        sdfs.append(sdf_edt_upscaled_averaging(image, factor=2.0, num_thresholds=8))

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(image_atlas, cmap="gray")

        for item in sdfs:
            plt.figure()
            plt.subplot(1, 2, 1)
            plot_sdf_contours(item, ref=image)
            plt.subplot(1, 2, 2)
            plot_sdf_error(item, ref=sdf_hq)

        plt.show()


if __name__ == "__main__":
    main()
