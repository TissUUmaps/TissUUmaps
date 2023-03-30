import copy
import json
import logging
import os

import h5py
import numpy as np
import pyvips
from scipy import sparse


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


tmap_template = {
    "layers": [{"name": "tissue.tif", "tileSource": "./img/tissue.tif.dzi"}],
    "markerFiles": [],
    "plugins": [],
    "collectionMode": True,
}


def getPalette(adata):
    palette = {}
    for uns in list(adata.get("uns", [])):
        if "_colors" in uns:
            uns_name = uns.replace("_colors", "")

            try:
                new_palette = dict(
                    zip(
                        adata.get(f"/obs/{uns_name}/categories").asstr()[...],
                        [x[:7] for x in adata.get(f"uns/{uns}/").asstr()[...]],
                    )
                )
                palette = dict(palette, **new_palette)
            except:
                pass
    return palette


def getVarList(adata):
    var = adata.get("/var")
    if "_index" in var.attrs.keys():
        varIndex = str(adata.get("/var").attrs["_index"])
        varList = [x.decode("utf-8") for x in adata.get(f"/var/{varIndex}", [])]
    else:
        # TODO
        varList = []
    return varList


def getObsList(adata):
    try:
        # if obs is a dataframe:
        obsList = list(adata.get("obs").dtype.names)
    except:
        # if obs is NOT a dataframe:
        obsList = list(adata.get("obs"))
    if "_index" in obsList:
        obsList = [obs for obs in obsList if obs != "_index"]

    return obsList


def to_csc_sparse(adata):
    if "encoding-type" in adata.get("X").attrs.keys():
        encodingTypeKey = "encoding-type"
    elif "h5sparse_format" in adata.get("X").attrs.keys():
        encodingTypeKey = "h5sparse_format"

    if "shape" in adata.get("X").attrs.keys():
        shapeKey = "shape"
    elif "h5sparse_shape" in adata.get("X").attrs.keys():
        shapeKey = "h5sparse_shape"

    if adata.get("X").attrs[encodingTypeKey] == "csr_matrix":
        logging.info(
            "Converting sparse CSR matrix to sparse CSC. This can take a long time."
        )
        shape = adata.get("X").attrs[shapeKey]
        X_data = adata.get("X/data")[:]
        X_indices = adata.get("X/indices")[:]
        X_indptr = adata.get("X/indptr")[:]
        csr_matrix = sparse.csr_matrix((X_data, X_indices, X_indptr), shape=shape)
        csc_matrix = sparse.csc_matrix(csr_matrix)

        del adata["X/indptr"]
        del adata["X/data"]
        del adata["X/indices"]
        adata["X/indptr"] = csc_matrix.indptr
        adata["X/data"] = csc_matrix.data
        adata["X/indices"] = csc_matrix.indices
        adata.get("X").attrs["encoding-type"] = "csc_matrix"


def get_write_adata(adata, path, basedir):
    path_out = os.path.splitext(path)[0] + "_tmap.h5ad"
    adata_out = h5py.File(os.path.join(basedir, path_out), "w")
    for obj in adata.keys():
        adata.copy(obj, adata_out)
    adata.close()
    adata = adata_out
    return adata, path_out


def h5ad_to_tmap(basedir, path, library_id=None):
    write_adata = False
    path_out = os.path.splitext(path)[0] + "_tmap.h5ad"
    # if os.path.isfile(os.path.join(basedir, path_out)):
    #    if os.path.getmtime(os.path.join(basedir, path)) < os.path.getmtime(
    #        os.path.join(basedir, path_out)
    #    ):
    #        path = path_out

    adata = h5py.File(os.path.join(basedir, path), "r")
    outputFolder = os.path.join(basedir, path) + "_files"
    relOutputFolder = os.path.basename(path) + "_files"

    img_key = "hires"

    markerScale = 1
    plugins = []

    globalX, globalY = "", ""
    for coordinates in [
        "spatial",
        "X_spatial",
        "X_umap",
        "tSNE",
    ]:
        if coordinates in list(adata.get("obsm", [])):
            globalX, globalY = f"/obsm/{coordinates};0", f"/obsm/{coordinates};1"
            break

    if list(adata.get("/obs/__categories/", [])) != []:
        if not write_adata:
            write_adata = True
            adata, path = get_write_adata(adata, path, basedir)

        for categ in list(adata.get("/obs/__categories/", [])):
            if categ in list(adata["/obs/"]):
                newgroup = adata.create_group(f"/obs/__{categ}__")
                newgroup.attrs["encoding-type"] = "categorical"
                adata.copy(f"/obs/{categ}", f"/obs/__{categ}__/codes")
                adata.copy(f"/obs/__categories/{categ}", f"/obs/__{categ}__/categories")
                del adata[f"/obs/{categ}"]
                adata["obs"].move(f"__{categ}__", f"{categ}")
        del adata[f"/obs/__categories"]

    layers = []
    library_ids = list(adata.get("uns/spatial", []))
    try:
        library_ids = adata["/obs/library_id/categories"].asstr()[...]
    except:
        try:
            library_ids = adata["/obs/__categories/library_id"].asstr()[...]
        except:
            pass

    coord_factor = 1
    for library_id in library_ids:
        coord_factor = float(
            adata.get(
                f"/uns/spatial/{library_id}/scalefactors/tissue_{img_key}_scalef", 1
            )[()]
        )
        os.makedirs(os.path.join(outputFolder, str(library_id), "img"), exist_ok=True)
        outputImage = os.path.join(outputFolder, str(library_id), "img", "tissue.tif")
        relOutputImage = os.path.join(
            relOutputFolder, str(library_id), "img", "tissue.tif"
        ).replace("\\", "/")
        layers.append({"name": library_id, "tileSource": relOutputImage + ".dzi"})
        if os.path.isfile(outputImage):
            continue
        try:
            img = np.array(adata.get(f"/uns/spatial/{library_id}/images/{img_key}"))

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

            logging.error(traceback.format_exc())

    use_libraries = len(library_ids) > 1

    library_col = ""
    if use_libraries:
        if "/obsm/spatial;" in globalX:
            try:
                library_codes_array = adata["/obs/library_id/codes"][...]
                library_categ_array = adata["/obs/library_id/categories"].asstr()[...]
                library_col = "/obs/library_id/codes"
            except:
                library_codes_array = adata["/obs/library_id"][...]
                library_categ_array = adata["/obs/__categories/library_id"][...]
                library_col = "/obs/library_id"

            if not "spatial_hires" in list(adata.get("/obsm", [])):
                if not write_adata:
                    write_adata = True
                    adata, path = get_write_adata(adata, path, basedir)
                spatial_array = adata["/obsm/spatial"][()]

                spatial_scaled_array = np.ones(spatial_array.shape)
                for library_index, library_id in enumerate(library_categ_array):
                    scale_factor = float(
                        adata.get(
                            f"/uns/spatial/{library_id}/scalefactors/tissue_{img_key}_scalef",
                            1,
                        )[()]
                    )
                    spatial_scaled_array[library_codes_array == library_index] = (
                        spatial_array[library_codes_array == library_index]
                        * scale_factor
                    )
                adata.create_dataset("/obsm/spatial_hires", data=spatial_scaled_array)
            globalX = "/obsm/spatial_hires;0"
            globalY = "/obsm/spatial_hires;1"
            coord_factor = 1

    if "spatial_connectivities" in list(adata.get("obsp", [])):
        spatial_connectivities = "/obsp/spatial_connectivities;join"
    else:
        spatial_connectivities = ""

    encodingType = None
    if "encoding-type" in adata.get("X").attrs.keys():
        encodingType = "encoding-type"
    elif "h5sparse_format" in adata.get("X").attrs.keys():
        encodingType = "h5sparse_format"
    if encodingType:
        if adata.get("X").attrs[encodingType] == "csr_matrix":
            if not write_adata:
                write_adata = True
                adata, path = get_write_adata(adata, path, basedir)

            to_csc_sparse(adata)

    varList = getVarList(adata)
    obsList = getObsList(adata)
    palette = getPalette(adata)

    new_tmap_project = copy.deepcopy(tmap_template)

    new_tmap_project["layers"] = layers
    new_tmap_project["plugins"] = plugins
    if "tmap" in list(adata.get("uns", [])):
        new_tmap_project = json.loads(
            adata.get(
                f"/uns/tmap",
                "{}",
            )[()]
        )
        if "markerFiles" not in new_tmap_project.keys():
            new_tmap_project["markerFiles"] = []
    new_tmap_project["markerFiles"].append(
        {
            "expectedHeader": {
                "X": globalX,
                "Y": globalY,
                "cb_cmap": "interpolateTurbo",
                "cb_col": "",
                "cb_gr_dict": "",
                "gb_col": "",
                "gb_name": "",
                "opacity": "1",
                "pie_col": "",
                "scale_col": "",
                "scale_factor": markerScale,
                "coord_factor": coord_factor,
                "shape_col": "",
                "shape_fixed": "disc",
                "shape_gr_dict": "",
                "edges_col": spatial_connectivities,
                "collectionItem_col": library_col,
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
                "sortby_check": True,
                "shape_col": False,
                "shape_fixed": True,
                "shape_gr": False,
                "shape_gr_dict": False,
                "shape_gr_rand": True,
                "collectionItem_col": use_libraries,
                "collectionItem_fixed": not use_libraries,
            },
            "hideSettings": True,
            "name": "Numerical observations",
            "path": os.path.basename(path),
            "dropdownOptions": [
                {
                    "optionName": obs,
                    "name": obs,
                    "expectedHeader.cb_col": f"/obs/{obs}",
                    "expectedHeader.sortby_col": f"/obs/{obs}",
                }
                for obs in obsList
                if adata.get(f"/obs/{obs}/categories") is None
                and adata.get(f"/obs/__categories/{obs}") is None
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
                "coord_factor": coord_factor,
                "shape_fixed": "disc",
                "edges_col": spatial_connectivities,
                "collectionItem_col": library_col,
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
                "sortby_check": False,
                "shape_gr": False,
                "shape_gr_dict": False,
                "shape_gr_rand": True,
                "collectionItem_col": use_libraries,
                "collectionItem_fixed": not use_libraries,
            },
            "hideSettings": True,
            "name": "Categorical observations",
            "path": os.path.basename(path),
            "dropdownOptions": [
                {"optionName": obs, "name": obs, "expectedHeader.gb_col": f"/obs/{obs}"}
                for obs in obsList
                if adata.get(f"/obs/{obs}/categories") is not None
                or adata.get(f"/obs/__categories/{obs}") is not None
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
                "cb_col": "",
                "scale_factor": markerScale,
                "coord_factor": coord_factor,
                "shape_fixed": "disc",
                "edges_col": spatial_connectivities,
                "collectionItem_col": library_col,
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
                "sortby_check": True,
                "shape_gr": False,
                "shape_gr_dict": False,
                "shape_gr_rand": True,
                "collectionItem_col": use_libraries,
                "collectionItem_fixed": not use_libraries,
            },
            "hideSettings": True,
            "name": "Gene expression",
            "path": os.path.basename(path),
            "dropdownOptions": [
                {
                    "optionName": gene,
                    "name": "Gene expression: " + gene,
                    "expectedHeader.cb_col": f"/X;{index}",
                    "expectedHeader.sortby_col": f"/X;{index}",
                }
                for index, gene in enumerate(varList)
            ],
            "title": "Gene expression",
            "uid": "mainTab",
        }
    )
    # with open(os.path.join(basedir, path + ".tmap"), "w") as f:
    #    json.dump(new_tmap_project, f, indent=4)
    adata.close()
    return new_tmap_project
