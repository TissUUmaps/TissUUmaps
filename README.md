# TissUUmaps - UU is for Uppsala University ;)
[TissUUmaps](https://tissuumaps.github.io/) is a browser-based tool for fast visualization and exploration of millions of data points overlaying a tissue sample. TissUUmaps can be used as a web service or locally in your computer, and allows users to share regions of interest and local statistics.

## Windows installation

1. Download the Windows Installer from [the last release](https://github.com/TissUUmaps/TissUUmaps/releases/latest) and install it. Note that the installer is not signed yet and may trigger warnings from the browser and from the firewall. You can safely pass these warnings.

2. Start TissUUmaps.

3. (Optional) add plugins to TissUUmaps by placing plugin files in the plugin folder. Default path is `C:\Program Files (x86)\TissUUmaps\plugins`. See [here](tissuumaps/plugins_available) for available plugins. 


## Installation from PIP

> Note that steps 1-4 are optional and can be replaced by installing a recent version of Python.

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) (or miniconda).
1. Create a new conda environment from an Anaconda prompt:
    ```bash
	conda create --name tissuumaps --channel conda-forge python=3.9
    ```

1. Activate the new conda environment:
    ```bash
	conda activate tissuumaps
    ```

1. You should be in the tissuumaps environment:
    ```bash
	(tissuumaps) C:\
    ```

1. Install the TissUUmaps library:
    ```bash
	pip install TissUUmaps[full]
    ```
1. (Optional) add plugins to TissUUmaps by placing plugin files in the plugin folder. Path (depending on your Anaconda installation) is: 
    ```bash
	{ANACONDA_INSTALL_PATH}\envs\tissuumaps\Lib\site-packages\tissuumaps\plugins` 
    ```
    See [here](tissuumaps/plugins_available) for available plugins.


### Option 1: Start the Graphical User Interface of TissUUmaps

1. Start the TissUUmaps user interface:
    ```bash
	tissuumaps
    ```

### Option 2: Start only the flask server

1. Start TissUUmaps as a server:
    ```bash
	tissuumaps_server path_to_your_images
    ```

1. Open http://127.0.0.1:5000/ in your favorite browser.

   > :warning: Remember that Flask is running on a built-in development server (`flask run`) and should not be used in production. If you want to deploy FlaskTissUUmaps on a production server, please read https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/ or any similar tutorial.

## Image format
TissUUmaps allows to visualize all images from a folder and sub-folders in the TissUUmaps viewer. By using a minimal deepzoom server, TissUUmaps removes the need for creating DZI files of every image.

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

TissUUmaps will convert any other format into a pyramidal tiff (in a temporary .tissuumaps folder) using [vips](https://github.com/libvips/libvips).
