import hashlib
import json
import os
import tempfile
from typing import Any, List, Literal, Optional, Union

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from flask import abort, make_response
from scipy.ndimage import zoom
from scipy.sparse import eye, spmatrix, vstack
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.preprocessing import normalize

COLORS = [
    [0.9019607843137255, 0.09803921568627451, 0.29411764705882354],
    [0.23529411764705882, 0.7058823529411765, 0.29411764705882354],
    [1.0, 0.8823529411764706, 0.09803921568627451],
    [0.2627450980392157, 0.38823529411764707, 0.8470588235294118],
    [0.9607843137254902, 0.5098039215686274, 0.19215686274509805],
    [0.5686274509803921, 0.11764705882352941, 0.7058823529411765],
    [0.27450980392156865, 0.9411764705882353, 0.9411764705882353],
    [0.9411764705882353, 0.19607843137254902, 0.9019607843137255],
    [0.7372549019607844, 0.9647058823529412, 0.047058823529411764],
    [0.9803921568627451, 0.7450980392156863, 0.7450980392156863],
    [0.0, 0.5019607843137255, 0.5019607843137255],
    [0.9019607843137255, 0.7450980392156863, 1.0],
    [0.6039215686274509, 0.38823529411764707, 0.1411764705882353],
    [1.0, 0.9803921568627451, 0.7843137254901961],
    [0.5019607843137255, 0.0, 0.0],
    [0.6666666666666666, 1.0, 0.7647058823529411],
    [0.5019607843137255, 0.5019607843137255, 0.0],
    [1.0, 0.8470588235294118, 0.6941176470588235],
    [0.0, 0.0, 0.4588235294117647],
    [0.5019607843137255, 0.5019607843137255, 0.5019607843137255],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0],
]

COLORS = [[int(255 * v) for v in RGB] for RGB in COLORS]


def polygons2json(polygons, cluster_class, cluster_names, colors=None):
    jsonROIs = []
    for i, polygon in enumerate(polygons):
        name = cluster_names[i]
        jsonROIs.append(
            {
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": []},
                "properties": {
                    "name": name,
                    "classification": {"name": cluster_class},
                    "color": colors[i] if colors is not None else [255, 0, 0],
                    "isLocked": False,
                },
            }
        )
        jsonROIs[-1]["geometry"]["coordinates"].append(polygon)
    return jsonROIs


def binary_mask_to_polygon(
    binary_mask: np.ndarray,
    tolerance: float = 0,
    offset: float = None,
    scale: float = None,
):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    from skimage.measure import approximate_polygon, find_contours

    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    binary_mask = zoom(binary_mask, 3, order=0, grid_mode=True)
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = find_contours(padded_binary_mask, 0.5)
    contours = [c - 1 for c in contours]
    # contours = np.subtract(contours, 1)
    for contour in contours:
        contour = approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = contour / 3
        contour = np.rint(contour)
        if scale is not None:
            contour = contour * scale
        if offset is not None:
            contour = contour + offset  # .ravel().tolist()

        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        polygons.append(contour.tolist())

    return polygons


def labelmask2geojson(
    labelmask: np.ndarray,
    region_name: str = "My regions",
    scale: float = 1.0,
    offset: float = 0,
):
    from skimage.measure import regionprops

    nclusters = np.max(labelmask)
    colors = [COLORS[k % len(COLORS)] for k in range(nclusters)]

    # Make JSON
    polygons = []
    cluster_names = [f"Region {l+1}" for l in np.arange(1, nclusters + 1)]
    for index, region in enumerate(regionprops(labelmask)):
        # take regions with large enough areas
        contours = binary_mask_to_polygon(
            region.image,
            offset=scale * np.array(region.bbox[0:2]) + offset,
            scale=scale,
        )
        polygons.append(contours)
    json = polygons2json(polygons, region_name, cluster_names, colors=colors)
    return json


def map2numeric(data: np.ndarray) -> np.ndarray:
    map2numeric = {k: i for i, k in enumerate(np.unique(data))}
    return np.array([map2numeric[v] for v in data])


def create_features(
    xy: np.ndarray,
    labels: np.ndarray,
    unique_labels: np.ndarray,
    sigma: float,
    bin_width: Union[float, str, None],
    min_genes_per_bin: int,
):
    if isinstance(bin_width, str):
        if bin_width == "auto":
            bin_width = sigma

    grid_props = {}
    # Compute binning matrix
    if bin_width is not None:
        B, grid_props = spatial_binning_matrix(
            xy, box_width=bin_width, return_grid_props=True
        )
    else:
        B = eye(len(xy))
    B = B.astype("float32")
    # Find center of mass for each point
    xy = ((B @ xy) / (B.sum(axis=1))).A

    # Create attribute matrix (ngenes x nuniques)
    attributes, _ = attribute_matrix(labels, unique_labels)
    attributes = attributes.astype("bool")

    features, adj = kde_per_label(xy, B @ attributes, sigma, return_neighbors=True)

    # Compute bin size
    bin_size = features.sum(axis=1).A.flatten()
    good_bins = bin_size >= min_genes_per_bin
    features = normalize(features, norm="l1", axis=1)

    return dict(
        features=features,
        grid_props=grid_props,
        good_bins=good_bins,
        back_map=B.T.nonzero()[1],
    )


def predict(
    kmeans_model, features: spmatrix, good_bins: np.ndarray, back_map: np.ndarray
):
    clusters = np.zeros(features.shape[0], dtype="int") - 1
    clusters[good_bins] = kmeans_model.predict(features[good_bins, :])
    return clusters[back_map], clusters


def points2regions(
    xy: np.ndarray,
    labels: np.ndarray,
    sigma: float,
    n_clusters: int,
    bin_width: Union[float, str, None] = "auto",
    min_genes_per_bin: int = 0,
    library_id_column: Union[np.ndarray, None] = None,
    convert_to_geojson: bool = False,
    seed: int = 42,
    region_name: str = "My regions",
):
    print(
        "xy",
        xy,
        "labels",
        labels,
        "sigma",
        sigma,
        "n_clusters",
        n_clusters,
        "bin_width",
        bin_width,
        "min_genes_per_bin",
        min_genes_per_bin,
        "library_id_column",
        library_id_column,
        "convert_to_geojson",
        convert_to_geojson,
        "seed",
        seed,
    )
    xy = np.array(xy, dtype="float32")

    # Iterate data by library ids
    if library_id_column is not None:
        unique_library_id = np.unique(library_id_column)
        iterdata = [
            (
                lib_id,
                (
                    xy[library_id_column == lib_id],
                    labels[library_id_column == lib_id],
                ),
            )
            for lib_id in unique_library_id
        ]
        get_slice = lambda library_id, data: data == library_id
    else:
        iterdata = [("id", (xy, labels))]
        get_slice = lambda library_id, data: np.ones(len(data), dtype="bool")

    unique_labels = np.unique(labels)
    results = {
        library_id: create_features(
            xy_slice, labels_slice, unique_labels, sigma, bin_width, min_genes_per_bin
        )
        for library_id, (xy_slice, labels_slice) in iterdata
    }

    # Create train features
    X_train = vstack([r["features"][r["good_bins"]] for r in results.values()])

    # Train K-Means
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    kmeans = kmeans.fit(X_train)

    # Predict
    for library_id, result_dict in results.items():
        cluster_per_gene, cluster_per_bin = predict(
            kmeans,
            result_dict["features"],
            result_dict["good_bins"],
            result_dict["back_map"],
        )
        results[library_id]["cluster_per_gene"] = cluster_per_gene
        results[library_id]["cluster_per_bin"] = cluster_per_bin

    # Add clusters to dataframe
    output_column = np.zeros(len(xy), dtype="int")
    for library_id in results.keys():
        if library_id_column is not None:
            library_id_slice_ind = get_slice(library_id, library_id_column)
        else:
            library_id_slice_ind = get_slice(library_id, xy)
        output_column[library_id_slice_ind] = results[library_id]["cluster_per_gene"]

    if convert_to_geojson:
        geojsons = []
        for result in results.values():
            grid_props = result["grid_props"]
            clusters = result["cluster_per_bin"]
            label_mask = np.zeros(grid_props["grid_size"], dtype="uint8")
            label_mask[tuple(ind for ind in grid_props["grid_coords"])] = clusters + 1
            label_mask = label_mask
            geojson = labelmask2geojson(
                label_mask,
                region_name=region_name,
                scale=1.0 / grid_props["grid_scale"],
                offset=grid_props["grid_offset"],
            )
            geojsons.append(geojson)
        return (output_column, geojsons)
    else:
        return (output_column, None)


from typing import Any, List, Literal, Optional, Union

from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.preprocessing import OneHotEncoder


def connectivity_matrix(
    xy: np.ndarray,
    method="knn",
    k: int = 5,
    r: Optional[float] = None,
    include_self: bool = False,
) -> sp.spmatrix:
    """
    Compute the connectivity matrix of a dataset based on either k-NN or radius search.

    Parameters
    ----------
    xy : np.ndarray
        The input dataset, where each row is a sample point.
    method : str, optional (default='knn')
        The method to use for computing the connectivity.
        Can be either 'knn' for k-nearest-neighbors or 'radius' for radius search.
    k : int, optional (default=5)
        The number of nearest neighbors to use when method='knn'.
    r : float, optional (default=None)
        The radius to use when method='radius'.
    include_self : bool, optional (default=False)
        If the matrix should contain self connectivities.

    Returns
    -------
    A : sp.spmatrix
        The connectivity matrix, with ones in the positions where two points are
            connected.
    """
    if method == "knn":
        A = kneighbors_graph(xy, k, include_self=include_self).astype("bool")
    else:
        A = radius_neighbors_graph(xy, r, include_self=include_self).astype("bool")
    return A


def attribute_matrix(
    cat: np.ndarray,
    unique_cat: Union[np.ndarray, Literal["auto"]] = "auto",
    return_encoder: bool = False,
):
    """
    Compute the attribute matrix from categorical data, based on one-hot encoding.

    Parameters
    ----------
    cat : np.ndarray
        The categorical data, where each row is a sample and each column is a feature.
    unique_cat : np.ndarray
        Unique categorical data used to setup up the encoder. If "auto", unique
        categories are automatically determined from cat.
    return_encoder : bool, optional (default=False)
        Whether to return the encoder object, in addition to the attribute matrix and
        categories list.

    Returns
    -------
    y : sp.spmatrix
        The attribute matrix, in sparse one-hot encoding format.
    categories : list
        The categories present in the data, as determined by the encoder.
    encoder : OneHotEncoder
        The encoder object, only returned if \`return_encoder\` is True.
    """
    X = np.array(cat).reshape((-1, 1))
    if not isinstance(unique_cat, str):
        unique_cat_list: Union[List[np.ndarray], Literal["auto"]] = [
            np.array(unique_cat)
        ]
    elif unique_cat == "auto":
        unique_cat_list = "auto"
    else:
        raise ValueError("\`unique_cat\` must be a numpy array or the string \`auto\`.")
    encoder = OneHotEncoder(
        categories=unique_cat_list, sparse_output=True, handle_unknown="ignore"
    )
    encoder.fit(X)
    y = encoder.transform(X)
    categories = list(encoder.categories_[0])
    if return_encoder:
        return y, categories, encoder
    return y, categories


def spatial_binning_matrix(
    xy: np.ndarray, box_width: float, return_grid_props: bool = False
) -> sp.spmatrix:
    """
    Compute a sparse matrix that indicates which points in a point cloud fall in which
    hyper-rectangular bins.

    Parameters:
    points (numpy.ndarray): An array of shape (N, D) containing the D-dimensional
        coordinates of N points in the point cloud.
    box_width (float): The width of the bins in which to group the points.

    Returns:
    sp.spmatrix: A sparse matrix of shape (num_bins, N) where num_bins is the number of
        bins. The matrix is such that the entry (i,j) is 1 if the j-th point falls in
        the i-th bin, and 0 otherwise.

    Example:
    >>> points = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2, 2, 2]])
    >>> bin_matrix = spatial_binning_matrix(points, 1)
    >>> print(bin_matrix.toarray())
    [[1 1 0 0]
        [1 1 0 0]
        [0 0 1 1]]
    """

    # Compute shifted coordinates
    mi, ma = xy.min(axis=0, keepdims=True), xy.max(axis=0, keepdims=True)
    xys = xy - mi

    # Compute grid size
    grid = ma - mi
    grid = grid.flatten()

    # Compute bin index
    bin_ids = xys // box_width
    bin_ids = bin_ids.astype("int")
    bin_ids = tuple(x for x in bin_ids.T)

    # Compute grid size in indices
    size = grid // box_width + 1
    size = tuple(x for x in size.astype("int"))

    # Convert bin_ids to integers
    linear_ind = np.ravel_multi_index(bin_ids, size)

    # Create a matrix indicating which markers fall in what bin
    bin_matrix, linear_unique_bin_ids = attribute_matrix(linear_ind)

    bin_matrix = bin_matrix.T

    if return_grid_props:
        sub_unique_bin_ids = np.unravel_index(linear_unique_bin_ids, size)
        grid_props = dict(
            grid_coords=sub_unique_bin_ids,
            grid_size=size,
            grid_offset=mi.flatten(),
            grid_scale=1.0 / box_width,
        )

    return (bin_matrix, grid_props) if return_grid_props else bin_matrix


def kde_per_label(
    xy: np.ndarray, features: sp.spmatrix, sigma: float, return_neighbors: bool = False
):
    """
    Computes the kernel density estimation (KDE) for each label in \`labels\`, using the
    data points in \`xy\` as inputs. Returns the KDE values as an attribute matrix, and
    the unique labels found in \`labels\`.

    Parameters:
    -----------
    xy : numpy.ndarray
        A 2D numpy array of shape (n, 2) containing the x-y coordinates of the data
        points.
    features : sp.spmatrix
        Features that are to be blured using KDE
    sigma : float
        The standard deviation of the Gaussian kernel to use in the KDE.

    Returns:
    --------
    Tuple of two numpy.ndarray:
        - \`att\`: A 2D numpy array of shape (n_labels, n_features), where n_labels is the
            number of unique labels in \`labels\`
                  and n_features is the number of attributes (columns) in \`labels\`. Each
                  row represents the KDE values for a single label.
        - \`unique_labels\`: A 1D numpy array containing the unique labels found in
            \`labels\`.
    """
    adj = connectivity_matrix(xy, method="radius", r=2.0 * sigma, include_self=True)
    row, col = adj.nonzero()
    d2 = (xy[row, 0] - xy[col, 0]) ** 2
    d2 = d2 + (xy[row, 1] - xy[col, 1]) ** 2
    d2 = np.exp(-d2 / (2 * sigma * sigma))
    aff = sp.csr_matrix((d2, (row, col)), shape=adj.shape, dtype="float32")
    if not return_neighbors:
        return aff @ features
    else:
        return aff @ features, adj


def local_maxima(A: sp.spmatrix, attributes: np.ndarray):
    # Convert to list of list
    L = A + sp.eye(A.shape[0])
    L = L.tolil()

    largest_neighbor = [np.max(attributes[n]) for n in L.rows]
    maximas = set({})
    neighbors = [set(n) for n in L.rows]

    # Loop over each node
    visited = set({})
    for node in np.flip(np.argsort(attributes)):
        if node not in visited:
            if attributes[node] >= largest_neighbor[node]:
                maximas.add(node)
                visited.update(neighbors[node])
            visited.add(node)

    maximas_arr = np.array(list(maximas))
    maximas_values = attributes[maximas_arr]
    maximas_arr = maximas_arr[np.flip(np.argsort(maximas_values))]
    return maximas_arr


def distance_filter(xy):
    from scipy.spatial import cKDTree as KDTree
    from sklearn.mixture import BayesianGaussianMixture

    # Find distance to 10th nearest neighbor
    distances, _ = KDTree(xy).query(xy, k=11)
    distances = distances[:, -1].reshape((-1, 1))
    mdl = BayesianGaussianMixture(n_components=2)
    mdl.fit(distances)
    means = mdl.means_
    fg_label = np.argmin(means.flatten())
    labels = mdl.predict(distances)
    fg_index = np.where(labels == fg_label)[0]
    return fg_index


class Plugin:
    def __init__(self, app):
        self.app = app

    def getCacheFile(self, jsonParam):
        strCache = self.app.basedir + json.dumps(jsonParam, sort_keys=True, indent=2)
        hashKey = hashlib.md5(strCache.encode("utf-8")).hexdigest()
        return os.path.join(
            tempfile.gettempdir(),
            "_".join([tempfile.gettempprefix(), "tmapPoints2Regions", hashKey]),
        )

    def checkServer(self, jsonParam):
        try:
            import os
            from typing import Union

            import h5py
            import numpy as np
            import scipy.sparse as sp
            from scipy.ndimage import zoom
            from scipy.sparse import eye, spmatrix, vstack
            from sklearn.cluster import MiniBatchKMeans as KMeans
            from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
            from sklearn.preprocessing import OneHotEncoder, normalize

            return make_response({"return": "success"})

        except:
            import traceback

            return make_response(
                {
                    "return": "error",
                    "message": "Impossible to run on server: \n"
                    + traceback.format_exc(),
                }
            )

    def Points2Regions(self, jsonParam):
        if not jsonParam:
            logging.error("No arguments, aborting.")
            abort(500)

        cacheFile = self.getCacheFile(jsonParam)
        # if os.path.isfile(cacheFile):
        #    with open(cacheFile, "r") as f:
        #        return make_response(f.read())
        if jsonParam["filetype"] != "h5":
            csvPath = os.path.abspath(
                os.path.join(self.app.basedir, jsonParam["csv_path"])
            )

            df = pd.read_csv(csvPath)
            xy = df[[jsonParam["xKey"], jsonParam["yKey"]]].to_numpy()
            labels = df[jsonParam["clusterKey"]].to_numpy()
        else:
            h5Path = os.path.abspath(
                os.path.join(self.app.basedir, jsonParam["csv_path"])
            )
            with h5py.File(h5Path, "r") as f:
                if ";" not in jsonParam["xKey"]:
                    xKey = jsonParam["xKey"]
                    x = f[xKey]
                else:
                    xKey = jsonParam["xKey"].split(";")
                    x = f[xKey[0]][()][:, int(xKey[1])]
                if ";" not in jsonParam["yKey"]:
                    yKey = jsonParam["yKey"]
                    y = f[yKey]
                else:
                    yKey = jsonParam["yKey"].split(";")
                    y = f[yKey[0]][()][:, int(yKey[1])]
                xy = np.stack((x, y), axis=1)
                try:
                    categories = f.get(jsonParam["clusterKey"] + "/categories")[()]
                    codes = f.get(jsonParam["clusterKey"] + "/codes")[()]
                    labels = categories[codes]
                except:
                    labels = f.get(jsonParam["clusterKey"])[()]
                labels = labels.astype(str)
        bins_per_res = float(jsonParam["bins_per_res"])
        sigma = float(jsonParam["sigma"])
        nclusters = int(jsonParam["nclusters"])
        expression_threshold = float(jsonParam["expression_threshold"])
        seed = int(jsonParam["seed"])
        region_name = jsonParam["region_name"]
        format = jsonParam["format"]
        stride = sigma / bins_per_res

        if format == "GeoJSON polygons":
            compute_regions = True
        else:
            compute_regions = False
        c, r = points2regions(
            xy,
            labels,
            sigma,
            nclusters,
            stride,
            expression_threshold,
            None,
            compute_regions,
            seed,
            region_name,
        )

        if format == "GeoJSON polygons":
            strOutput = json.dumps(r)
        else:
            strOutput = np.array2string(c, separator=",", threshold=c.shape[0])

        with open(cacheFile, "w") as f:
            f.write(strOutput)
        resp = make_response(strOutput)
        return resp


# Points2Regions.setMessage("Run failed.")
# data = dict(dataUtils.data.object_entries())
# data_obj = data[Points2Regions.get("_dataset")]
# processeddata = dict(data_obj._processeddata.object_entries())
# x_field = data_obj._X
# y_field = data_obj._Y

# x = np.asarray(processeddata[x_field].to_py())
# y = np.asarray(processeddata[y_field].to_py())
# if (data_obj._collectionItem_col in processeddata.keys()):
#     lib_id = np.asarray(processeddata[data_obj._collectionItem_col].to_py())
#     x += lib_id * max(x) * 1.1
# xy = np.vstack((x,y)).T

# labels = np.asarray(processeddata[Points2Regions.get("_clusterKey")].to_py())
# from os.path import join

# Points2Regions.setMessage("Run failed.")
# stride = float(Points2Regions.get("_stride"))
# sigma = float(Points2Regions.get("_sigma"))
# nclusters = int(Points2Regions.get("_nclusters"))
# expression_threshold = float(Points2Regions.get("_expression_threshold"))
# seed = int(Points2Regions.get("_seed"))
# region_name = Points2Regions.get("_region_name")

# c,r = points2regions(
#     xy,
#     labels,
#     sigma * stride,
#     nclusters,
#     stride,
#     expression_threshold,
#     None,
#     True,
#     seed,
#     region_name
#     )
# import json
# print (json.dumps(r))
# if (Points2Regions.get("_format")== "GeoJSON polygons"):
#     Points2Regions.loadRegions(json.dumps(r))
# else:
#     Points2Regions.loadClusters(to_js(c))
# Points2Regions.setMessage("")
