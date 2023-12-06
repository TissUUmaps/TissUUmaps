/**
 * @file Points2Regions.js
 * @author Axel Andersson, Christophe Avenel
 */

/**
 * @namespace Points2Regions
 * @classdesc The root namespace for Points2Regions.
 */
var Points2Regions;
Points2Regions = {
  name: "Points2Regions Plugin",
  parameters: {
    _nclusters: {
      label: "Number of clusters (default 8):",
      type: "number",
      default: 8,
    },
    _expression_threshold: {
      label: "Min points per bin (increase to avoid regions with few markers):",
      type: "number",
      default: 1,
    },
    _sigma: {
      label:
        "Spatial resolution (increase/decrease for coarser/finer regions):",
      type: "number",
      default: 100,
    },
    _selectStride: {
      label: "Select spatial resolution on tissue (optional)",
      type: "button",
    },
    _run: {
      label: "Run Points2Regions",
      type: "button",
    },
    _downloadCSV: {
      label: "Download data as CSV",
      type: "button",
    },
    _advancedSection: {
      label: "Only change these settings if you know what you are doing!",
      title: "Advanced settings",
      type: "section",
    },
    _refresh: {
      label: "Refresh drop-down lists based on loaded markers",
      type: "button",
    },
    _dataset: {
      label: "Select marker dataset:",
      type: "select",
      default: "lol",
    },
    _clusterKey: {
      label: "Select Points2Regions Key:",
      type: "select",
    },
    _bins_per_res: {
      label: "Number of bins per `resolution` (default 3):",
      type: "number",
      default: 3,
    },
    _seed: {
      label: "Random seed (used during KMeans):",
      type: "number",
      default: 0,
      attributes: { step: 1 },
    },
    _format: {
      label: "Output regions as",
      type: "select",
      default: "New label per marker",
      options: ["GeoJSON polygons", "New label per marker"],
    },
    _server: {
      label: "Run Points2Regions on the server",
      type: "checkbox",
      default: true,
    },
  },
  _region_name: "Clusters",
};

/**
 * @summary */
Points2Regions.init = function (container) {
  Points2Regions.inputTrigger("_refresh");
  Points2Regions.container = container;
  // Points2Regions.initPython()
  Points2Regions._api(
    "checkServer",
    null,
    function (data) {
      if (data["return"] == "error") {
        Points2Regions.set("_server", false);
        Points2Regions.initPython();
        let serverCheckBoxID = Points2Regions.getInputID("_server");
        let serverCheckbox = document.getElementById(serverCheckBoxID);
        serverCheckbox.disabled = true;
        serverCheckbox.parentElement.title = data["message"];
        var tooltip = new bootstrap.Tooltip(serverCheckbox.parentElement, {
          placement: "right",
        });
        tooltip.enable();
      }
    },
    function () {
      Points2Regions.set("_server", false);
      Points2Regions.initPython();
      let serverCheckBoxID = Points2Regions.getInputID("_server");
      let serverCheckbox = document.getElementById(serverCheckBoxID);
      serverCheckbox.disabled = true;
      serverCheckbox.parentElement.title =
        "Unable to run on server, check that you have all dependencies installed (scikit-learn).";
      var tooltip = new bootstrap.Tooltip(serverCheckbox.parentElement, {
        placement: "right",
      });
      tooltip.enable();
    },
  );
  let advancedSectionIndex = 7;

  let advancedSectionElement = document.querySelector(
    `#plugin-Points2Regions div:nth-child(${advancedSectionIndex}) div h6`,
  );
  advancedSectionElement?.setAttribute("data-bs-toggle", "collapse");
  advancedSectionElement?.setAttribute("data-bs-target", "#collapse_advanced");
  advancedSectionElement?.setAttribute("aria-expanded", "false");
  advancedSectionElement?.setAttribute("aria-controls", "collapse_advanced");
  advancedSectionElement?.setAttribute(
    "class",
    "collapse_button_transform border-bottom-0 p-1 collapsed",
  );
  advancedSectionElement?.setAttribute("style", "cursor: pointer;");
  advancedSectionElement?.setAttribute("title", "Click to expand");
  let newDiv = document.createElement("div");
  newDiv.setAttribute("id", "collapse_advanced");
  newDiv.setAttribute("class", "collapse");
  $("#plugin-Points2Regions").append(newDiv);
  let advancedSectionSubtitle = document.querySelector(
    `#plugin-Points2Regions div:nth-child(${advancedSectionIndex}) div p`,
  );
  newDiv.appendChild(advancedSectionSubtitle);
  for (
    let indexElement = advancedSectionIndex + 1;
    indexElement < Object.keys(Points2Regions.parameters).length + 1;
    indexElement++
  ) {
    let element = document.querySelector(
      `#plugin-Points2Regions div:nth-child(${advancedSectionIndex + 1})`,
    );
    newDiv.appendChild(element);
  }
};

Points2Regions.run = function () {
  if (Points2Regions.get("_server")) {
    var csvFile = dataUtils.data[Points2Regions.get("_dataset")]._csv_path;
    if (typeof csvFile === "object") {
      interfaceUtils.alert(
        "This plugin can only run on datasets generated from buttons. Please convert your dataset to a button (Markers > Advanced Options > Generate button from tab)",
      );
      return;
    }
    // Get the path from url:
    if (dataUtils.data[Points2Regions.get("_dataset")]._filetype != "h5") {
      const path = dataUtils.getPath();
      if (path != null) {
        csvFile = path + "/" + csvFile;
      }
    }
    loadingModal = interfaceUtils.loadingModal(
      "Points2Regions... Please wait.",
    );
    $.ajax({
      type: "post",
      url: "/plugins/Points2Regions/Points2Regions",
      contentType: "application/json; charset=utf-8",
      data: JSON.stringify({
        xKey: dataUtils.data[Points2Regions.get("_dataset")]._X,
        yKey: dataUtils.data[Points2Regions.get("_dataset")]._Y,
        clusterKey: Points2Regions.get("_clusterKey"),
        nclusters: Points2Regions.get("_nclusters"),
        expression_threshold: Points2Regions.get("_expression_threshold"),
        sigma: Points2Regions.get("_sigma"),
        bins_per_res: Points2Regions.get("_bins_per_res"),
        region_name: Points2Regions.get("_region_name"),
        seed: Points2Regions.get("_seed"),
        format: Points2Regions.get("_format"),
        csv_path: csvFile,
        filetype: dataUtils.data[Points2Regions.get("_dataset")]._filetype,
      }),
      success: function (data) {
        if (Points2Regions.get("_format") == "GeoJSON polygons") {
          Points2Regions.loadRegions(data);
        } else {
          console.log(data);
          data = data.substring(1, data.length - 1);
          let clusters = data.split(",").map(function (x) {
            return parseInt(x);
          });
          console.log(clusters);
          Points2Regions.loadClusters(clusters);
        }
        setTimeout(function () {
          $(loadingModal).modal("hide");
        }, 500);
      },
      complete: function (data) {
        // do something, not critical.
      },
      error: function (data) {
        console.log("Error:", data);
        setTimeout(function () {
          $(loadingModal).modal("hide");
        }, 500);
        interfaceUtils.alert(
          "Error during Points2Regions, check logs. This plugin only works on a pip installation of TissUUmaps, with the extra packages: pandas, sklearn, skimage",
        );
      },
    });
  } else {
    var content = `
from typing import Union
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.preprocessing import normalize
from scipy.sparse import eye, vstack, spmatrix
from scipy.ndimage import zoom
import numpy as np

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

def binary_mask_to_polygon(binary_mask: np.ndarray, tolerance: float=0, offset: float=None, scale: float=None):
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
    contours = [c-1 for c in contours]
    #contours = np.subtract(contours, 1)
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
    region_name:str="My regions",
    scale:float=1.0,
    offset:float = 0
):
    from skimage.measure import regionprops

    nclusters = np.max(labelmask)
    colors = [COLORS[k % len(COLORS)] for k in range(nclusters)]

    # Make JSON
    polygons = []
    cluster_names = [f"Region {l+1}" for l in np.arange(1,nclusters+1)]
    for index, region in enumerate(regionprops(labelmask)):
        # take regions with large enough areas
        contours = binary_mask_to_polygon(
            region.image,
            offset=scale*np.array(region.bbox[0:2]) + offset,
            scale=scale
        )
        polygons.append(contours)
    json = polygons2json(polygons, region_name, cluster_names, colors=colors)
    return json


def map2numeric(data: np.ndarray) -> np.ndarray:
    map2numeric = {k : i for i,k in enumerate(np.unique(data))}
    return np.array([map2numeric[v] for v in data])


def create_features(xy: np.ndarray, labels: np.ndarray, unique_labels:np.ndarray, sigma: float, bin_width: Union[float, str, None], min_genes_per_bin: int):
    if isinstance(bin_width, str):
        if bin_width == 'auto':
            bin_width = sigma

    grid_props = {}
    # Compute binning matrix
    if bin_width is not None:
        B, grid_props = spatial_binning_matrix(xy, box_width=bin_width, return_grid_props=True)
    else:
        B = eye(len(xy))
    B = B.astype('float32')
    # Find center of mass for each point
    xy = ((B @ xy) / (B.sum(axis=1))).A

    # Create attribute matrix (ngenes x nuniques)
    attributes, _ = attribute_matrix(labels, unique_labels)
    attributes = attributes.astype('bool')

    features, adj = kde_per_label(xy, B @ attributes, sigma, return_neighbors = True)

    # Compute bin size
    bin_size = features.sum(axis=1).A.flatten()
    good_bins = bin_size >= min_genes_per_bin
    features = normalize(features, norm='l1', axis=1)

    return dict(
        features=features,
        grid_props=grid_props,
        good_bins=good_bins,
        back_map=B.T.nonzero()[1]
    )


def predict(kmeans_model, features: spmatrix, good_bins: np.ndarray, back_map: np.ndarray):
    clusters = np.zeros(features.shape[0], dtype='int') - 1
    clusters[good_bins] = kmeans_model.predict(features[good_bins,:])
    return clusters[back_map], clusters

def points2regions(xy: np.ndarray, labels: np.ndarray, sigma: float, n_clusters: int, bin_width: Union[float, str, None] = 'auto', min_genes_per_bin:int = 0, library_id_column: Union[np.ndarray,None] = None, convert_to_geojson: bool = False, seed:int=42, region_name:str="My regions"):
    print (
        "xy",xy, "labels", labels,"sigma",sigma, "n_clusters", n_clusters, "bin_width", bin_width, "min_genes_per_bin", min_genes_per_bin, "library_id_column", library_id_column, "convert_to_geojson", convert_to_geojson, "seed", seed
    )
    xy = np.array(xy, dtype="float32")

    # Iterate data by library ids
    if library_id_column is not None:
        unique_library_id = np.unique(library_id_column)
        iterdata = [
            (lib_id, (
                xy[library_id_column==lib_id],
                labels[library_id_column==lib_id]
            )) for lib_id in unique_library_id
        ]
        get_slice = lambda library_id, data: data == library_id
    else:
        iterdata = [('id', (xy, labels))]
        get_slice = lambda library_id, data: np.ones(len(data), dtype='bool')


    unique_labels = np.unique(labels)
    results = {
        library_id : create_features(
            xy_slice,
            labels_slice,
            unique_labels,
            sigma,
            bin_width,
            min_genes_per_bin
        )
        for library_id, (xy_slice, labels_slice) in iterdata
    }

    # Create train features
    X_train = vstack([
        r['features'][r['good_bins']] for r in results.values()
    ])

    # Train K-Means
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed)
    kmeans = kmeans.fit(X_train)

    # Predict
    for library_id, result_dict in results.items():
        cluster_per_gene, cluster_per_bin = predict(kmeans, result_dict['features'], result_dict['good_bins'], result_dict['back_map'])
        results[library_id]['cluster_per_gene'] = cluster_per_gene
        results[library_id]['cluster_per_bin'] = cluster_per_bin

    # Add clusters to dataframe
    output_column = np.zeros(len(xy), dtype='int')
    for library_id in results.keys():
        if library_id_column is not None:
            library_id_slice_ind = get_slice(library_id, library_id_column)
        else:
            library_id_slice_ind = get_slice(library_id, xy)
        output_column[library_id_slice_ind] = results[library_id]['cluster_per_gene']


    if convert_to_geojson:
        geojsons = []
        for result in results.values():
            grid_props = result['grid_props']
            clusters = result['cluster_per_bin']
            label_mask = np.zeros(grid_props['grid_size'], dtype='uint8')
            label_mask[tuple(ind for ind in grid_props['grid_coords'])] = clusters + 1
            label_mask = label_mask
            geojson = labelmask2geojson(label_mask, region_name=region_name, scale=1.0/grid_props['grid_scale'], offset=grid_props['grid_offset'])
            geojsons.append(geojson)
        return (output_column, geojsons)
    else:
        return (output_column, None)


from typing import Any, List, Literal, Optional, Union

import scipy.sparse as sp
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
        A = kneighbors_graph(xy, k, include_self=include_self).astype('bool')
    else:
        A = radius_neighbors_graph(xy, r, include_self=include_self).astype('bool')
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
            grid_scale=1.0/box_width
        )

    return (bin_matrix, grid_props) if return_grid_props else bin_matrix

def kde_per_label(xy: np.ndarray, features: sp.spmatrix, sigma: float, return_neighbors: bool = False):
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
    d2 = (xy[row,0] - xy[col,0])**2
    d2 = d2 + (xy[row,1] - xy[col,1])**2
    d2 = np.exp(-d2 / (2 * sigma * sigma))
    aff = sp.csr_matrix((d2, (row, col)), shape=adj.shape, dtype='float32')
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



from js import dataUtils
from js import Points2Regions
from pyodide.ffi import to_js
import numpy as np

Points2Regions.setMessage("Run failed.")
data = dict(dataUtils.data.object_entries())
data_obj = data[Points2Regions.get("_dataset")]
processeddata = dict(data_obj._processeddata.object_entries())
x_field = data_obj._X
y_field = data_obj._Y

x = np.asarray(processeddata[x_field].to_py(), dtype="float32")
y = np.asarray(processeddata[y_field].to_py(), dtype="float32")
if (data_obj._collectionItem_col in processeddata.keys()):
    lib_id = np.asarray(processeddata[data_obj._collectionItem_col].to_py())
else:
    lib_id = None
xy = np.vstack((x,y)).T

labels = np.asarray(processeddata[Points2Regions.get("_clusterKey")].to_py())
from os.path import join

Points2Regions.setMessage("Run failed.")
bins_per_res = float(Points2Regions.get("_bins_per_res"))
sigma = float(Points2Regions.get("_sigma"))
nclusters = int(Points2Regions.get("_nclusters"))
expression_threshold = float(Points2Regions.get("_expression_threshold"))
seed = int(Points2Regions.get("_seed"))
region_name = Points2Regions.get("_region_name")
stride = sigma / bins_per_res

if (Points2Regions.get("_format")== "GeoJSON polygons"):
    compute_regions = True
else:
    compute_regions = False

c,r = points2regions(
    xy,
    labels,
    sigma,
    nclusters,
    stride,
    expression_threshold,
    lib_id,
    compute_regions,
    seed,
    region_name
    )
import json
print (json.dumps(r))
if (Points2Regions.get("_format")== "GeoJSON polygons"):
    Points2Regions.loadRegions(json.dumps(r))
else:
    Points2Regions.loadClusters(to_js(c))
Points2Regions.setMessage("")

`;
    if (Points2Regions.get("_dataset") === "") {
      Points2Regions.set("_dataset", Object.keys(dataUtils.data)[0]);
    }
    if (Points2Regions.get("_clusterKey") === undefined) {
      Points2Regions.set(
        "_clusterKey",
        dataUtils.data[Points2Regions.get("_dataset")]._gb_col,
      );
    }
    Points2Regions.setMessage("Running Python code...");
    setTimeout(() => {
      Points2Regions.executePythonString(content);
    }, 10);
  }
};

Points2Regions.inputTrigger = function (parameterName) {
  if (parameterName == "_refresh") {
    interfaceUtils.cleanSelect(Points2Regions.getInputID("_dataset"));
    interfaceUtils.cleanSelect(Points2Regions.getInputID("_clusterKey"));

    var datasets = Object.keys(dataUtils.data).map(function (e, i) {
      return {
        value: e,
        innerHTML: document.getElementById(e + "_tab-name").value,
      };
    });
    interfaceUtils.addObjectsToSelect(
      Points2Regions.getInputID("_dataset"),
      datasets,
    );
    var event = new Event("change");
    interfaceUtils
      .getElementById(Points2Regions.getInputID("_dataset"))
      .dispatchEvent(event);
    if (dataUtils.data[Points2Regions.get("_dataset")]._processeddata) {
      Points2Regions.estimateBinSize();
    }
  } else if (parameterName == "_dataset") {
    if (!dataUtils.data[Points2Regions.get("_dataset")]) return;
    interfaceUtils.cleanSelect(Points2Regions.getInputID("_clusterKey"));
    interfaceUtils.addElementsToSelect(
      Points2Regions.getInputID("_clusterKey"),
      dataUtils.data[Points2Regions.get("_dataset")]._csv_header,
    );
    Points2Regions.set(
      "_clusterKey",
      dataUtils.data[Points2Regions.get("_dataset")]._gb_col,
    );

    if (dataUtils.data[Points2Regions.get("_dataset")]._filetype == "h5") {
      select311 = interfaceUtils._mGenUIFuncs.intputToH5(
        Points2Regions.get("_dataset"),
        interfaceUtils.getElementById(Points2Regions.getInputID("_clusterKey")),
      );
      Points2Regions.set(
        "_clusterKey",
        dataUtils.data[Points2Regions.get("_dataset")]._gb_col,
      );
      select311.addEventListener("change", (event) => {
        Points2Regions.set("_clusterKey", select311.value);
      });
    }
  } else if (parameterName == "_selectStride") {
    Points2Regions.selectStride();
  } else if (parameterName == "_run") {
    Points2Regions.run();
  } else if (parameterName == "_downloadCSV") {
    Points2Regions.downloadCSV();
  } else if (parameterName == "_server") {
    if (!Points2Regions.get("_server")) {
      Points2Regions.initPython();
    }
  }
};

Points2Regions.estimateBinSize = function (parameterName) {
  /*
  let data_obj = dataUtils.data[Points2Regions.get("_dataset")];
  let XKey = dataUtils.data[Points2Regions.get("_dataset")]._X;
  let YKey = dataUtils.data[Points2Regions.get("_dataset")]._Y;
  let X = data_obj._processeddata[XKey];
  let Y = data_obj._processeddata[YKey];
  let width = Quartile_75(X) - Quartile_25(X);
  let height = Quartile_75(Y) - Quartile_25(Y);
  let bin_width = 2 * width * X.length ** (-1 / 3);
  let bin_height = 2 * height * Y.length ** (-1 / 3);
  console.log(width, height, bin_width, bin_height, X.length, Y.length);
  Points2Regions.set("_stride", (bin_width + bin_height) / 2);
  */
};

Points2Regions.selectStride = function (parameterName) {
  var startSelection = null;
  var pressHandler = function (event) {
    console.log("Pressed!");
    var OSDviewer = tmapp["ISS_viewer"];
    startSelection = OSDviewer.viewport.pointFromPixel(event.position);
  };
  var moveHandler = function (event) {
    if (startSelection == null) return;
    let OSDviewer = tmapp["ISS_viewer"];

    let normCoords = OSDviewer.viewport.pointFromPixel(event.position);
    let tiledImage = OSDviewer.world.getItemAt(0);
    let rectangle = tiledImage.viewportToImageRectangle(
      new OpenSeadragon.Rect(
        startSelection.x,
        startSelection.y,
        normCoords.x - startSelection.x,
        normCoords.y - startSelection.y,
      ),
    );
    let canvas =
      overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
    let regionobj = d3
      .select(canvas)
      .append("g")
      .attr("class", "_stride_region");
    let elements = document.getElementsByClassName("stride_region");
    for (let element of elements) element.parentNode.removeChild(element);

    if (rectangle.width <= 0) {
      return;
    }
    let width = Math.max(
      normCoords.x - startSelection.x,
      normCoords.y - startSelection.y,
    );
    console.log(width, normCoords.x, normCoords.y);
    let polyline = regionobj
      .append("rect")
      .attr("width", width)
      .attr("height", width)
      .attr("x", startSelection.x)
      .attr("y", startSelection.y)
      .attr("fill", "#ADD8E6")
      .attr("stroke", "#ADD8E6")
      .attr("fill-opacity", 0.3)
      .attr("stroke-opacity", 0.7)
      .attr("stroke-width", 0.002 / tmapp["ISS_viewer"].viewport.getZoom())
      .attr(
        "stroke-dasharray",
        0.004 / tmapp["ISS_viewer"].viewport.getZoom() +
          "," +
          0.004 / tmapp["ISS_viewer"].viewport.getZoom(),
      )
      .attr("class", "stride_region");
    Points2Regions.set("_sigma", rectangle.width);
    return;
  };
  var dragHandler = function (event) {
    event.preventDefaultAction = true;
  };
  var releaseHandler = function (event) {
    console.log("Released!", pressHandler, releaseHandler, dragHandler);
    startSelection = null;
    tmapp["ISS_viewer"].removeHandler("canvas-press", pressHandler);
    tmapp["ISS_viewer"].removeHandler("canvas-release", releaseHandler);
    tmapp["ISS_viewer"].removeHandler("canvas-drag", dragHandler);
    var elements = document.getElementsByClassName("stride_region");
    for (var element of elements) element.parentNode.removeChild(element);
  };
  tmapp["ISS_viewer"].addHandler("canvas-press", pressHandler);
  tmapp["ISS_viewer"].addHandler("canvas-release", releaseHandler);
  tmapp["ISS_viewer"].addHandler("canvas-drag", dragHandler);
  new OpenSeadragon.MouseTracker({
    element: tmapp["ISS_viewer"].canvas,
    moveHandler: (event) => moveHandler(event),
  }).setTracking(true);
};

Points2Regions.loadClusters = function (data) {
  let data_obj = dataUtils.data[Points2Regions.get("_dataset")];
  data_obj._processeddata["Points2Regions"] = data;
  data_obj._gb_col = "Points2Regions";
  data_obj._processeddata.columns.push("Points2Regions");
  interfaceUtils.addElementsToSelect(
    Points2Regions.get("_dataset") + "_gb-col-value",
    ["Points2Regions"],
  );
  document.getElementById(
    Points2Regions.get("_dataset") + "_gb-col-value",
  ).value = "Points2Regions";
  let colors = {
    "-1": "#000000",
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#279e68",
    3: "#d62728",
    4: "#aa40fc",
    5: "#8c564b",
    6: "#e377c2",
    7: "#b5bd61",
    8: "#17becf",
    9: "#aec7e8",
    10: "#ffbb78",
    11: "#98df8a",
    12: "#ff9896",
    13: "#c5b0d5",
    14: "#c49c94",
    15: "#f7b6d2",
    16: "#dbdb8d",
    17: "#9edae5",
    18: "#ad494a",
    19: "#8c6d31",
  };
  document
    .getElementById(Points2Regions.get("_dataset") + "_cb-bygroup-dict")
    .click();
  document.getElementById(
    Points2Regions.get("_dataset") + "_cb-bygroup-dict-val",
  ).value = JSON.stringify(colors);
  dataUtils._quadtreesLastInputs = null;
  glUtils._markerInputsCached[Points2Regions.get("_dataset")] = null;
  dataUtils.updateViewOptions(Points2Regions.get("_dataset"));
};

Points2Regions.loadRegions = function (data) {
  // Change stroke width for computation reasons:
  regionUtils._polygonStrokeWidth = 0.0005;
  groupRegions = Object.values(regionUtils._regions)
    .filter((x) => x.regionClass == Points2Regions._region_name)
    .forEach(function (region) {
      regionUtils.deleteRegion(region.id);
    });

  regionsobj = JSON.parse(data)[0];
  console.log(regionsobj);
  regionUtils.JSONValToRegions(regionsobj);
  $("#title-tab-regions").tab("show");
  $(
    document.getElementById("regionClass-" + Points2Regions._region_name),
  ).collapse("show");
  $("#" + Points2Regions._region_name + "_group_fill_ta").click();
};

/*
 * Only helper functions below
 *
 */
Points2Regions.executePythonString = function (text) {
  // prepare objects exposed to Python

  // pyscript
  let div = document.createElement("div");
  let html = `
        <py-script>
  ${text}
        </py-script>
        `;
  div.innerHTML = html;

  // if we did this before, remove the script from the body
  if (Points2Regions.myPyScript) {
    Points2Regions.myPyScript.remove();
  }
  // now remember the new script
  Points2Regions.myPyScript = div.firstElementChild;
  try {
    // add it to the body - this will already augment the tag in certain ways
    document.body.appendChild(Points2Regions.myPyScript);
    // execute the code / evaluate the expression
    //Points2Regions.myPyScript.evaluate();
  } catch (error) {
    console.error("Python error:");
    console.error(error);
  }
};

Points2Regions.initPython = function () {
  if (!document.getElementById("pyScript")) {
    Points2Regions.setMessage("Loading Python interpreter...");
    var link = document.createElement("link");
    link.src = "https://pyscript.net/latest/pyscript.css";
    link.id = "pyScript";
    link.rel = "stylesheet";
    document.head.appendChild(link);

    var script = document.createElement("script");
    script.src = "https://pyscript.net/latest/pyscript.js";
    script.defer = true;
    document.head.appendChild(script);

    var pyconfig = document.createElement("py-config");
    pyconfig.innerHTML = "packages=['scikit-learn','scikit-image']";
    document.head.appendChild(pyconfig);
    Points2Regions.executePythonString(`
        from js import Points2Regions
        Points2Regions.pythonLoaded()
      `);
  }
};

Points2Regions.pythonLoaded = function () {
  Points2Regions.setMessage("");
};

Points2Regions.setMessage = function (text) {
  if (!document.getElementById("Points2Regions_message")) {
    var label_row = HTMLElementUtils.createRow({});
    var label_col = HTMLElementUtils.createColumn({ width: 12 });
    var label = HTMLElementUtils.createElement({
      kind: "p",
      id: "Points2Regions_message",
    });
    label.setAttribute("class", "badge bg-warning text-dark");
    label_row.appendChild(label_col);
    label_col.appendChild(label);
    Points2Regions.container.appendChild(label_row);
  }
  document.getElementById("Points2Regions_message").innerText = text;
};

Points2Regions.downloadCSV = function () {
  var csvRows = [];
  let alldata = dataUtils.data[Points2Regions.get("_dataset")]._processeddata;
  let headers = Object.keys(alldata);
  headers.splice(headers.indexOf("columns"), 1);
  csvRows.push(headers.join(","));
  let zip = (...rows) => [...rows[0]].map((_, c) => rows.map((row) => row[c]));
  let rows = zip(...headers.map((h) => alldata[h]));
  const escape = (text) =>
    text.replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/,/g, "\\,");

  //let escaped_array = rows.map(fields => fields.map(escape))
  let csv =
    headers.toString() +
    "\n" +
    rows.map((fields) => fields.join(",")).join("\n");

  regionUtils.downloadPointsInRegionsCSV(csv);
};

Points2Regions._api = function (endpoint, data, success, error) {
  $.ajax({
    // Post select to url.
    type: "post",
    url: "/plugins/Points2Regions" + "/" + endpoint,
    contentType: "application/json; charset=utf-8",
    data: JSON.stringify(data),
    success: function (data) {
      success(data);
    },
    complete: function (data) {
      // do something, not critical.
    },
    error: error
      ? error
      : function (data) {
          interfaceUtils.alert(
            data.responseText.replace("\n", "<br/>"),
            "Error on the plugin's server response:",
          );
        },
  });
};

//adapted from https://blog.poettner.de/2011/06/09/simple-statistics-with-php/

function Quartile_25(data) {
  return Quartile(data, 0.25);
}

function Quartile_75(data) {
  return Quartile(data, 0.75);
}

function Quartile(data, q) {
  data = Array_Sort_Numbers(data);
  var pos = (data.length - 1) * q;
  var base = Math.floor(pos);
  var rest = pos - base;
  if (data[base + 1] !== undefined) {
    return data[base] + rest * (data[base + 1] - data[base]);
  } else {
    return data[base];
  }
}

function Array_Sort_Numbers(inputarray) {
  var sortedarray = inputarray.slice(0);
  return sortedarray.sort(function (a, b) {
    return a - b;
  });
}

function Array_Sum(t) {
  return t.reduce(function (a, b) {
    return a + b;
  }, 0);
}

function Array_Average(data) {
  return Array_Sum(data) / data.length;
}

function Array_Stdev(tab) {
  var i,
    j,
    total = 0,
    mean = 0,
    diffSqredArr = [];
  for (i = 0; i < tab.length; i += 1) {
    total += tab[i];
  }
  mean = total / tab.length;
  for (j = 0; j < tab.length; j += 1) {
    diffSqredArr.push(Math.pow(tab[j] - mean, 2));
  }
  return Math.sqrt(
    diffSqredArr.reduce(function (firstEl, nextEl) {
      return firstEl + nextEl;
    }) / tab.length,
  );
}
