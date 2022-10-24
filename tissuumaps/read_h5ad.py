import copy
import logging
import os
import string
from enum import Enum
from typing import Mapping, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import pyvips


class Empty(Enum):
    token = 0


_empty = Empty.token

# Helpers
def _check_spatial_data(
    uns: Mapping, library_id: Union[Empty, None, str]
) -> Tuple[Optional[str], Optional[Mapping]]:
    """
    Given a mapping, try and extract a library id/ mapping with spatial data.

    Assumes this is `.uns` from how we parse visium data.
    """
    spatial_mapping = uns.get("spatial", {})
    if library_id is _empty:
        if len(spatial_mapping) > 1:
            raise ValueError(
                "Found multiple possible libraries in `.uns['spatial']. Please specify."
                f" Options are:\n\t{list(spatial_mapping.keys())}"
            )
        elif len(spatial_mapping) == 1:
            library_id = list(spatial_mapping.keys())[0]
        else:
            library_id = None
    if library_id is not None:
        spatial_data = spatial_mapping[library_id]
    else:
        spatial_data = None
    return library_id, spatial_data


def _check_img(
    spatial_data: Optional[Mapping],
    img: Optional[np.ndarray],
    img_key: Union[None, str, Empty],
    bw: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Resolve image for spatial plots.
    """
    if img is None and spatial_data is not None and img_key is _empty:
        img_key = next(
            (k for k in ["hires", "lowres"] if k in spatial_data["images"]),
        )  # Throws StopIteration Error if keys not present
    if img is None and spatial_data is not None and img_key is not None:
        img = spatial_data["images"][img_key]
    if bw:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img, img_key


def _check_crop_coord(
    crop_coord: Optional[tuple],
    scale_factor: float,
) -> Tuple[float, float, float, float]:
    """Handle cropping with image or basis."""
    if crop_coord is None:
        return None
    if len(crop_coord) != 4:
        raise ValueError("Invalid crop_coord of length {len(crop_coord)}(!=4)")
    crop_coord = tuple(c * scale_factor for c in crop_coord)
    return crop_coord


def _check_scale_factor(
    spatial_data: Optional[Mapping],
    img_key: Optional[str],
    scale_factor: Optional[float],
) -> float:
    """Resolve scale_factor, defaults to 1."""
    if scale_factor is not None:
        return scale_factor
    elif spatial_data is not None and img_key is not None:
        if img_key == "raw":
            return 1
        return spatial_data["scalefactors"][f"tissue_{img_key}_scalef"]
    else:
        return 1.0


def numpy2vips(a):
    dtype_to_format = {
        "uint8": "uchar",
        "int8": "char",
        "uint16": "ushort",
        "int16": "short",
        "uint32": "uint",
        "int32": "int",
        "float32": "float",
        "float64": "double",
        "complex64": "complex",
        "complex128": "dpcomplex",
    }
    try:
        height, width, bands = a.shape
    except:
        height, width = a.shape
        bands = 1
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(
        linear.data, width, height, bands, dtype_to_format[str(a.dtype)]
    )
    return vi


def to_filename(key):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return "".join(c if c in valid_chars else " " for c in key)


import os

import numpy as np

tmap_template = {
    "layers": [{"name": "tissue.tif", "tileSource": "./img/tissue.tif.dzi"}],
    "markerFiles": [],
    "plugins": [],
    "compositeMode": "collection",
}

h5ad_cache = {}


def read_h5ad(filename, cache=True):
    filename = filename.replace("\\", "/")
    if not cache and filename in h5ad_cache.keys():
        del h5ad_cache[filename]
    if filename in h5ad_cache.keys():
        logging.info("H5AD file in cache: " + filename)
        adata = h5ad_cache[filename]
    else:
        logging.info("H5AD file not in cache: loading " + filename)
        adata = anndata.read_h5ad(filename, backed="r")
        # Cleaning var / obs names that can not be in file names:
        logging.info("Cleaning var and obs names to store as files")
        adata.var_names = [to_filename(key) for key in adata.var_names]
        for key in adata.obs.keys():
            clean_key = to_filename(key)
            if key != clean_key:
                adata.obs[clean_key] = adata.obs[key]
                del adata.obs[key]
        h5ad_cache[filename] = adata
        logging.info("Loading AnnData object done.")
    return adata


def h5ad_obs_to_csv(basedir, path, obsName):
    adata = read_h5ad(os.path.join(basedir, path))
    img_key = "hires"
    outputFolder = os.path.join(basedir, path) + "_files"

    obsdata = adata.obs[[obsName]].copy()  # .reset_index(drop=True)
    obsdata.columns = ["obs"]
    obsdata["library_id"] = ""
    if "X_umap" in adata.obsm:
        obsdata["umap_0"] = adata.obsm["X_umap"][:, 0]
        obsdata["umap_1"] = adata.obsm["X_umap"][:, 1]
    if "spatial" in adata.obsm:
        obsdata["globalX"] = adata.obsm["spatial"][:, 0]
        obsdata["globalY"] = adata.obsm["spatial"][:, 1]
    if "spatial_connectivities" in adata.obsp:
        logging.info("Found spatial neighborhood graph!")
        matrix = adata.obsp["spatial_connectivities"]  # Sparse matrix in CSR format
        edges = [np.where(matrix[i].toarray())[1] for i in range(matrix.shape[0])]
        for i in range(0, matrix.shape[0]):
            # Convert edge list for row into string with indices separated by ";"
            edges[i] = (
                str(list(edges[i])).replace(",", ";").strip("[").strip("]").strip(" ")
            )
        obsdata["obsp"] = edges
        logging.info("Spatial neighborhood graph added to CSV data")

    for library_index, library_id in enumerate(adata.uns.get("spatial", {})):
        library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
        scale_factor = _check_scale_factor(
            spatial_data, img_key=img_key, scale_factor=None
        )
        if "spatial" in adata.obsm:
            if "library_id" in adata.obs:
                obsdata["globalX"][adata.obs.library_id == library_id] *= scale_factor
                obsdata["globalY"][adata.obs.library_id == library_id] *= scale_factor
                obsdata["library_id"][
                    adata.obs.library_id == library_id
                ] = library_index
            else:
                obsdata["globalX"] *= scale_factor
                obsdata["globalY"] *= scale_factor

    obsdata.reset_index(drop=True).to_csv(
        os.path.join(outputFolder, "csv", "obs", obsName + ".csv")
    )


def h5ad_var_to_csv(basedir, path, obsName):
    adata = read_h5ad(os.path.join(basedir, path))
    img_key = "hires"
    outputFolder = os.path.join(basedir, path) + "_files"

    geneExp = pd.DataFrame(adata[:, obsName].X.toarray())
    geneExp.columns = ["gene_expression"]
    geneExp["library_id"] = ""
    if "X_umap" in adata.obsm:
        geneExp["umap_0"] = adata.obsm["X_umap"][:, 0]
        geneExp["umap_1"] = adata.obsm["X_umap"][:, 1]
    if "spatial" in adata.obsm:
        geneExp["globalX"] = adata.obsm["spatial"][:, 0]
        geneExp["globalY"] = adata.obsm["spatial"][:, 1]
    for library_index, library_id in enumerate(adata.uns.get("spatial", {})):
        library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
        scale_factor = _check_scale_factor(
            spatial_data, img_key=img_key, scale_factor=None
        )
        if "spatial" in adata.obsm:
            if "library_id" in adata.obs:
                lib_index = adata[:, obsName].obs.library_id == library_id
                geneExp["globalX"][lib_index.reset_index(drop=True)] *= scale_factor
                geneExp["globalY"][lib_index.reset_index(drop=True)] *= scale_factor
                geneExp["library_id"][lib_index.reset_index(drop=True)] = library_index
            else:
                geneExp["globalX"] *= scale_factor
                geneExp["globalY"] *= scale_factor

    geneExp.to_csv(os.path.join(outputFolder, "csv", "var", obsName + ".csv"))


def h5ad_to_tmap(basedir, path, library_id=None):
    adata = read_h5ad(os.path.join(basedir, path), cache=False)

    def is_numeric_dtype(object):
        try:
            return np.issubdtype(object.dtype, np.number)
        except:
            return False

    outputFolder = os.path.join(basedir, path) + "_files"
    relOutputFolder = os.path.basename(path) + "_files"
    os.makedirs(os.path.join(outputFolder, "csv/var"), exist_ok=True)
    os.makedirs(os.path.join(outputFolder, "csv/obs"), exist_ok=True)

    img_key = "hires"
    obsList = adata.obs.columns
    markerScale = 1
    plugins = []

    globalX, globalY = "", ""
    if "X_umap" in adata.obsm:
        globalX, globalY = "umap_0", "umap_1"
    if "spatial" in adata.obsm:
        globalX, globalY = "globalX", "globalY"

    palette = {}
    for uns in adata.uns:
        if "_colors" in uns:
            uns_name = uns.replace("_colors", "")
            try:
                _colors = dict(
                    zip(
                        [
                            str(i)
                            for i in range(len(adata.obs[uns_name].cat.categories))
                        ],
                        adata.uns[uns_name + "_colors"],
                    )
                )
                new_palette = {
                    adata.obs[uns_name].cat.categories[int(k)]: v[:7]
                    for k, v in _colors.items()
                }
                palette = dict(palette, **new_palette)
            except:
                pass

    # if library_id is not None:
    #    adata = adata[adata.obs.library_id == library_id].copy()
    # else:
    #    adata = adata.copy()

    # print ("library_id:", library_id)
    layers = []
    for library_id in adata.uns.get("spatial", {}):
        library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
        os.makedirs(os.path.join(outputFolder, str(library_id), "img"), exist_ok=True)
        outputImage = os.path.join(outputFolder, str(library_id), "img", "tissue.tif")
        relOutputImage = os.path.join(
            relOutputFolder, str(library_id), "img", "tissue.tif"
        ).replace("\\", "/")
        layers.append({"name": library_id, "tileSource": relOutputImage + ".dzi"})
        if os.path.isfile(outputImage):
            continue
        try:
            img = spatial_data["images"][img_key]

            if type(img) == str:
                img = pyvips.Image.new_from_file(img)
            else:
                if img.max() <= 1:
                    img *= 255
                img = numpy2vips(img)

            img.tiffsave(
                outputImage,
                pyramid=True,
                tile=True,
                tile_width=256,
                tile_height=256,
                compression="jpeg",
                Q=90,
                properties=True,
            )
        except:
            img = None
            import traceback

            print(traceback.format_exc())
    varList = [
        {"gene": gene, "display": gene} if type(gene) == str else gene
        for gene in adata.var_names
    ]

    new_tmap_project = copy.deepcopy(tmap_template)

    new_tmap_project["layers"] = layers
    new_tmap_project["plugins"] = plugins
    new_tmap_project["markerFiles"].append(
        {
            "expectedHeader": {
                "X": globalX,
                "Y": globalY,
                "cb_cmap": "interpolateTurbo",
                "cb_col": "obs",
                "cb_gr_dict": "",
                "gb_col": "",
                "gb_name": "null",
                "opacity": "1",
                "pie_col": "null",
                "scale_col": "null",
                "scale_factor": markerScale,
                "shape_col": "null",
                "shape_fixed": "disc",
                "shape_gr_dict": "",
                "collectionItem_col": "library_id",
                "collectionItem_fixed": "0",
            },
            "expectedRadios": {
                "cb_col": True,
                "cb_gr": False,
                "cb_gr_dict": True,
                "cb_gr_key": True,
                "cb_gr_rand": False,
                "pie_check": False,
                "scale_check": False,
                "shape_col": False,
                "shape_fixed": True,
                "shape_gr": False,
                "shape_gr_dict": False,
                "shape_gr_rand": True,
                "collectionItem_col": True,
                "collectionItem_fixed": False,
            },
            "hideSettings": True,
            "name": "Numerical observations",
            "path": [
                f"{relOutputFolder}/csv/obs/{obs}.csv"
                for obs in obsList
                if is_numeric_dtype(adata.obs[obs])
            ],
            "title": "Numerical observations",
            "uid": "mainTab",
        }
    )
    new_tmap_project["markerFiles"].append(
        {
            "expectedHeader": {
                "X": globalX,
                "Y": globalY,
                "cb_gr_dict": palette,
                "gb_col": "obs",
                "opacity": "1",
                "scale_factor": markerScale,
                "shape_fixed": "disc",
                "collectionItem_col": "library_id",
                "collectionItem_fixed": "0",
            },
            "expectedRadios": {
                "cb_col": False,
                "cb_gr": True,
                "cb_gr_dict": True,
                "cb_gr_key": False,
                "cb_gr_rand": False,
                "pie_check": False,
                "scale_check": False,
                "shape_col": False,
                "shape_fixed": True,
                "shape_gr": False,
                "shape_gr_dict": False,
                "shape_gr_rand": True,
                "collectionItem_col": True,
                "collectionItem_fixed": False,
            },
            "hideSettings": True,
            "name": "Categorical observations",
            "path": [
                f"{relOutputFolder}/csv/obs/{obs}.csv"
                for obs in obsList
                if not is_numeric_dtype(adata.obs[obs])
            ],
            "title": "Categorical observations",
            "uid": "mainTab",
        }
    )
    new_tmap_project["markerFiles"].append(
        {
            "expectedHeader": {
                "X": globalX,
                "Y": globalY,
                "cb_cmap": "interpolateViridis",
                "cb_col": "gene_expression",
                "scale_factor": markerScale,
                "shape_fixed": "disc",
                "collectionItem_col": "library_id",
                "collectionItem_fixed": "0",
            },
            "expectedRadios": {
                "cb_col": True,
                "cb_gr": False,
                "cb_gr_dict": False,
                "cb_gr_key": True,
                "cb_gr_rand": False,
                "pie_check": False,
                "scale_check": False,
                "shape_col": False,
                "shape_fixed": True,
                "shape_gr": False,
                "shape_gr_dict": False,
                "shape_gr_rand": True,
                "collectionItem_col": True,
                "collectionItem_fixed": False,
            },
            "hideSettings": True,
            "name": "Gene expression",
            "path": [
                f'{relOutputFolder}/csv/var/{gene["display"]}.csv' for gene in varList
            ],
            "title": "Gene expression",
            "uid": "mainTab",
        }
    )
    return new_tmap_project
