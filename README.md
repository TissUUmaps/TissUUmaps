# TissUUmaps - UU is for Uppsala University ;)

![TissUUmaps banner](https://github.com/TissUUmaps/TissUUmapsCore/blob/master/misc/design/logo-github-2443-473.png)

[TissUUmaps](https://tissuumaps.github.io/) is a browser-based tool for fast visualization and exploration of millions of data points overlaying a tissue sample. TissUUmaps can be used as a web service or locally in your computer, and allows users to share regions of interest and local statistics.

## Windows installation

1. Download the Windows Installer from [the last release](https://github.com/TissUUmaps/TissUUmaps/releases/latest) and install it. Note that the installer is not signed yet and may trigger warnings from the browser and from the firewall. You can safely pass these warnings.

## PIP installation (for Linux and Mac)

1. Install `libvips` for your system: [https://www.libvips.org/install.html](https://www.libvips.org/install.html)

    An easy way to install `libvips` is to use an [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) environment with `libvips`:
    ```bash
    conda create -y -n tissuumaps_env -c conda-forge python=3.9 libvips
    conda activate tissuumaps_env
    ```

1. Install the TissUUmaps library using `pip`:
    ```bash
    pip install "TissUUmaps[full]"
    ```

1. Start the TissUUmaps user interface:
    ```bash
    tissuumaps
    ```

1. Or start TissUUmaps as a local server:
    ```bash
    tissuumaps_server path_to_your_images
    ```
    And open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your favorite browser.

## Plugins
You can add plugins to TissUUmaps from the application menu, or by placing plugin files in the folder `[USER_FOLDER]/.tissuumaps/plugins/`. See [here](https://tissuumaps.github.io/TissUUmaps/plugins/) for available plugins. 


## Image format
TissUUmaps can read whole slide images in any format recognized by the [OpenSlide library](https://openslide.org/api/python/#openslide-python):
 * Aperio (.svs, .tif)
 * Hamamatsu (.ndpi, .vms, .vmu)
 * Leica (.scn)
 * MIRAX (.mrxs)
 * Philips (.tiff)
 * Sakura (.svslide)
 * Trestle (.tif)
 * Ventana (.bif, .tif)
 * Generic tiled TIFF (.tif)

TissUUmaps will convert any other format into a pyramidal tiff (in a temporary `.tissuumaps` folder) using [vips](https://github.com/libvips/libvips).

If your image fails to open, try converting it to `tif` format using an external tool.
