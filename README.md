# FlaskTissUUmaps
FlaskTissUUmaps is a minimal python server for [TissUUmaps](https://tissuumaps.research.it.uu.se/) using Flask that comes with a standalone User Interface.

## Differences with TissUUmaps

### Image format
FlaskTissUUmaps allows to visualize all images from a folder and sub-folders in TissUUmaps. By using a minimal deepzoom server, FlaskTissUUmaps removes the need for creating DZI files of every image.

FlaskTissUUmaps can read whole slide images in any format recognized by the [OpenSlide library](https://openslide.org/api/python/#openslide-python):
 * Aperio (.svs, .tif)
 * Hamamatsu (.ndpi, .vms, .vmu)
 * Leica (.scn)
 * MIRAX (.mrxs)
 * Philips (.tiff)
 * Sakura (.svslide)
 * Trestle (.tif)
 * Ventana (.bif, .tif)
 * Generic tiled TIFF (.tif)

plus classical images in any format recognized by the [PIL library](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html) (png, jpg, *etc.*)

> :warning: Warning: classical non pyramidal images will have to be loaded in RAM and will be read entirely for each generated tile. If you have big images, consider converting them in pyramidal format using VIPS.

### Saving TissUUmaps state
FlaskTissUUmaps allows to save all TissUUmaps states (Gene expressions, Cell morphology, Regions, Layers) so that you can reload images with all additional information, just as you saved them.



## Windows installation

1. Download the Windows Installer from [the last release](https://github.com/wahlby-lab/FlaskTissUUmaps/releases/latest) and install it. Note that the installer is not signed yet and may trigger warnings from the browser and from the firewall. You can safely pass these warnings.

2. Start TissUUmaps.

## Installation from PIP

> Note that steps 1-4 are optional and can be replaced by installing a recent version of Python.

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) (or miniconda).
1. Create a new conda environment from an Anaconda prompt:
    ```bash
	conda create --name tissuumaps
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
	pip install tissuumaps
    ```

## Option 1: Start the Graphical User Interface of TissUUmaps

1. Start the TissUUmaps user interface:
    ```bash
	tissuumaps
    ```

## Option 2: Start only the flask server

1. Start TissUUmaps as a server:
    ```bash
	tissuumaps_server path_to_your_images
    ```

1. Open http://127.0.0.1:5000/ in your favorite browser.

   > :warning: Remember that Flask is running on a built-in development server (`flask run`) and should not be used in production. If you want to deploy FlaskTissUUmaps on a production server, please read https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/ or any similar tutorial.

1. Options:

FlaskTissUUmaps can be used with the following options:
```bash
Usage: tissuumaps_server [options] [slide-directory]

Options:
  -h, --help            show this help message and exit
  -B, --ignore-bounds   display entire scan area
  -c FILE, --config=FILE
                        config file
  -d, --debug           run in debugging mode (insecure)
  -e PIXELS, --overlap=PIXELS
                        overlap of adjacent tiles [1]
  -f {jpeg|png}, --format={jpeg|png}
                        image format for tiles [jpeg]
  -l ADDRESS, --listen=ADDRESS
                        address to listen on [127.0.0.1]
  -p PORT, --port=PORT  port to listen on [5000]
  -Q QUALITY, --quality=QUALITY
                        JPEG compression quality [75]
  -s PIXELS, --size=PIXELS
                        tile size [254]
  -D LEVELS, --depth=LEVELS
                        folder depth search for opening files [4]
```

