import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.preprocessing import normalize
from typing import Optional


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


def attribute_matrix(cat: np.ndarray, unique_labels=None):
    """
    Compute the attribute matrix from categorical data, based on one-hot encoding.

    Parameters
    ----------
    cat : np.ndarray
        The categorical data, where each row is a sample and each column is a feature.
    unique_cat : np.ndarray
        Unique categorical data used to setup up the encoder. If "auto", unique
        categories are automatically determined from cat.

    Returns
    -------
    y : sp.spmatrix
        The attribute matrix, in sparse one-hot encoding format.
    categories : list
        The categories present in the data, as determined by the encoder.
    """
    if unique_labels is None:
        categories, col_ind = np.unique(cat, return_inverse=True)
    else:
        categories = unique_labels
        map_to_ind = {u: i for i, u in enumerate(categories)}
        col_ind = np.array([map_to_ind[i] for i in cat])

    row_ind = np.arange(len(cat))
    shape = (len(cat), len(categories))
    val = np.ones(len(cat), dtype=bool)
    y = sp.csr_matrix((val, (row_ind, col_ind)), shape=shape, dtype=bool)
    return y, categories


def spatial_binning_matrix(
    xy: np.ndarray, bin_width: float, return_grid_props: bool = False
) -> sp.spmatrix:
    # Compute shifted coordinates
    mi, ma = xy.min(axis=0, keepdims=True), xy.max(axis=0, keepdims=True)
    xys = xy - mi

    # Compute grid size
    grid = ma - mi
    grid = grid.flatten()

    # Compute bin index
    bin_ids = xys // bin_width
    bin_ids = bin_ids.astype("int")
    bin_ids = tuple(x for x in bin_ids.T)

    # Compute grid size in indices
    size = grid // bin_width + 1
    size = tuple(x for x in size.astype("int"))

    # Convert bin_ids to integers
    linear_ind = np.ravel_multi_index(bin_ids, size)

    # Create a matrix indicating which markers fall in what bin
    bin_matrix, linear_unique_bin_ids = attribute_matrix(linear_ind)

    bin_matrix = bin_matrix.T
    sub_unique_bin_ids = np.unravel_index(linear_unique_bin_ids, size)

    offset = mi.flatten()
    xy_bin = np.vstack(
        tuple(sub_unique_bin_ids[i] * bin_width + offset[i] for i in range(2))
    ).T

    bin_matrix = bin_matrix.astype("float32").tocsr()
    if return_grid_props:
        grid_props = dict(
            non_empty_bins=linear_unique_bin_ids,
            grid_coords=sub_unique_bin_ids,
            grid_size=size,
            grid_offset=mi.flatten(),
            grid_scale=1.0 / bin_width,
        )
        return bin_matrix, xy_bin, grid_props
    return bin_matrix, xy_bin


def create_neighbors_matrix(grid_size, non_empty_indices):
    # Total number of grid points
    n_grid_pts = grid_size[0] * grid_size[1]

    # Create 2D arrays of row and column indices for all grid points
    rows, cols = np.indices(grid_size)
    linear_indices = rows * grid_size[1] + cols

    # Convert linear indices of non-empty grid points to subindices
    non_empty_subindices = np.unravel_index(non_empty_indices, grid_size)

    # Create arrays representing potential neighbors in four directions
    neighbors_i = np.array([0, 0, -1, 1, 0])
    neighbors_j = np.array([-1, 1, 0, 0, 0])

    # Compute potential neighbors for all non-empty grid points
    neighbor_candidates_i = non_empty_subindices[0][:, np.newaxis] + neighbors_i
    neighbor_candidates_j = non_empty_subindices[1][:, np.newaxis] + neighbors_j

    # Filter out neighbors that are outside the grid
    valid_neighbors = np.where(
        (0 <= neighbor_candidates_i)
        & (neighbor_candidates_i < grid_size[0])
        & (0 <= neighbor_candidates_j)
        & (neighbor_candidates_j < grid_size[1])
    )

    # Create COO format data for the sparse matrix
    non_empty_indices = np.array(non_empty_indices)
    data = np.ones_like(valid_neighbors[0])
    rows = non_empty_indices[valid_neighbors[0]]
    cols = (
        neighbor_candidates_i[valid_neighbors],
        neighbor_candidates_j[valid_neighbors],
    )
    cols = cols[0] * grid_size[1] + cols[1]

    # Create the sparse matrix using COO format
    neighbors = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_grid_pts, n_grid_pts), dtype=bool
    )

    # Extract the submatrix for non-empty indices
    neighbors = neighbors[non_empty_indices, :][:, non_empty_indices]

    return neighbors


def find_inverse_distance_weights(ij, A, B, bin_width):
    # Create sparse matrix
    num_pts = A.shape[0]
    num_other_pts = B.shape[0]
    cols, rows = ij

    # Inverse distance weighing
    distances = np.linalg.norm(A[rows, :] - B[cols, :], axis=1)
    good_ind = distances <= bin_width * 1.000005

    vals = 1.0 / (distances + 1e-5)
    # vals = vals / vals.sum(axis=1, keepdims=True)
    vals = vals.flatten()
    # data = np.ones_like(rows, dtype=float)
    vals = vals[good_ind]
    rows = rows[good_ind]
    cols = cols[good_ind]
    sparse_matrix = sp.csr_matrix(
        (vals, (rows, cols)), shape=(num_pts, num_other_pts), dtype="float32"
    )
    sparse_matrix.eliminate_zeros()
    normalize(sparse_matrix, norm="l1", copy=False)
    return sparse_matrix


def inverse_distance_interpolation(
    xy: np.ndarray,
    labels: np.ndarray,
    unique_labels: np.ndarray,
    pixel_width: float,
    smooth: float,
    min_markers_per_pixel: int,
):
    # Create attribute matrix (nobs x n_unique_labels)
    attributes, _ = attribute_matrix(labels, unique_labels)

    # Number of resolution levels.
    num_levels = 4
    pixel_widths = np.linspace(pixel_width, pixel_width * smooth, num_levels)

    # B maps each gene to a pixel of the highest resolution (smallest pixel width)
    B, xy, grid_props = spatial_binning_matrix(
        xy, bin_width=pixel_width, return_grid_props=True
    )

    # Compute features (frequency of each label in each pixel)
    features = B.dot(attributes)

    # Compute center of each pixel
    bin_center_xy = xy + 0.5 * pixel_width

    # Compute lower resolution pixels
    # as well as weightes between neighboring bins
    Ws, Bs = [], []
    density = features.sum(axis=1)
    X = density.copy()

    for level in range(1, num_levels):
        Bi, xyi, props = spatial_binning_matrix(
            xy, bin_width=pixel_widths[level], return_grid_props=True
        )

        N = create_neighbors_matrix(props["grid_size"], props["non_empty_bins"])
        # Find a 4-connectivity graph that connects adjacent non-empty bins

        # Find which high-resolution pixels are connected to
        # what low-resolution pixel.
        neighbors = N.dot(Bi).nonzero()
        # Neighbors is of shape 2 x n
        # where the first row are indices to low-resolution pixels
        # and the second row are indices to high-resolution pixels

        # Find weights between the low and high resolution pixels
        low_res_pixel_center_xy = xyi + 0.5 * pixel_widths[level]

        W = find_inverse_distance_weights(
            neighbors,
            bin_center_xy,
            low_res_pixel_center_xy,
            pixel_widths[level],
        )

        # Append matrices for later use
        Ws.append(W)
        Bs.append(Bi)

        # Compute density
        density += W.dot(Bi.dot(X))

    # Remove bins with low density
    passed_threshold = density / num_levels >= min_markers_per_pixel
    features = features.multiply(passed_threshold)
    features.eliminate_zeros()

    # Compute features by aggregating different resolutions
    X = features.copy()

    for Wi, Bi in zip(Ws, Bs):
        features += Wi.dot(Bi.dot(X))

    # Prepare outputs
    # Convert to numpy array
    passed_threshold = passed_threshold.A.flatten()

    # Log data
    features.data = np.log1p(features.data)

    # Normalizing factor each bin
    s = features.sum(axis=1)

    # Normalize
    norms_r = 1.0 / (s + 1e-5)
    norms_r[np.isinf(norms_r)] = 0.0
    features = features.multiply(norms_r).tocsr()

    return dict(
        # Features (num_high_res_pixels x n_unique_markers)
        features=features,
        # Position of each pixel
        xy_pixel=xy,
        # Normalizing factor for each pixel
        norms=norms_r.A.flatten(),
        # Dictionary with grid properties.
        # Such as the shape of the grid
        grid_props=grid_props,
        # Whether a pixel (bin) passed the
        # density threshold
        passed_threshold=passed_threshold,
        # Sequence of indicies of length
        # nobs that indicates which high-res
        # pixel each observed marker belongs to.
        pix2marker_ind=B.T.nonzero()[1],
    )


import numpy as np
from skimage.measure import approximate_polygon, find_contours
from scipy.ndimage import zoom

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


def binarymask2polygon(
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

    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    binary_mask = zoom(binary_mask, 3, order=0, grid_mode=True)
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = find_contours(padded_binary_mask, 0.5)
    contours = [c - 1 for c in contours]
    for contour in contours:
        contour = approximate_polygon(contour, tolerance)
        contour = np.fliplr(contour)  # yx instead of x
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
    min_area=0,
    colors=None,
):
    from skimage.measure import regionprops

    nclusters = np.max(labelmask) + 1
    if colors is None:
        colors = [COLORS[k % len(COLORS)] for k in range(nclusters)]
    if isinstance(colors, np.ndarray):
        colors = np.round(colors * 255).astype("uint8")
        colors = colors.tolist()
    # Make JSON
    polygons = []
    cluster_names = [f"Region {l}" for l in np.arange(nclusters)]
    props = regionprops(labelmask + 1)
    cc = []
    for index, region in enumerate(props):
        # take regions with large enough areas
        if region.area > min_area:
            offset_yx = scale * np.flip(np.array(region.bbox[0:2])) + offset
            contours = binarymask2polygon(region.image, offset=offset_yx, scale=scale)
            cc.append(colors[region.label - 1])

            polygons.append(contours)
    features = polygons2json(polygons, region_name, cluster_names, colors=cc)
    geojson = {"type": "FeatureCollection", "features": features}
    return geojson


from typing import Optional, Any, Dict, List, Union, Tuple, Literal
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from skimage.color import lab2rgb
from scipy.spatial import cKDTree as KDTree
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.ndimage import distance_transform_edt as edt, binary_erosion


def _compute_cluster_center(cluster_labels, features):
    A, unique_labels = attribute_matrix(cluster_labels)
    average_features_per_cluster = ((A.T @ features) / (A.T.sum(axis=1))).A
    return unique_labels, average_features_per_cluster


def _merge_clusters(kmeans, merge_thresh):
    linkage_matrix = linkage(
        kmeans.cluster_centers_, method="complete", metric="cosine"
    )
    # fcluster returns in [1, n_clust], we prefer to start at 0 (hence -1).
    return fcluster(linkage_matrix, merge_thresh, criterion="maxclust") - 1


def _rgb_to_hex(rgb):
    # Convert each channel value from float (0-1) to integer (0-255)
    r, g, b = [int(x * 255) for x in rgb]
    # Format the RGB values as a hexadecimal string
    hex_color = f"#{r:02X}{g:02X}{b:02X}"
    return hex_color


class Points2RegionClass:
    """
    Points2Regions is a tool for clustering and defining regions based on categorical
    marker data, which are commonly encountered in spatial biology.


    """

    def __init__(
        self,
        xy: np.ndarray,
        labels: np.ndarray,
        pixel_width: float,
        pixel_smoothing: float,
        min_num_pts_per_pixel: float = 0.0,
        datasetids: Optional[np.ndarray] = None,
    ):
        """
        Initializes Points2Regions instance.

        Parameters
        ----------
        xy : np.ndarray
            Array of coordinates. Must be of shape (N x 2).
        labels : np.ndarray
            Array of labels. Must be of shape (N).
        pixel_width : float
            Width of a pixel in the rasterized image.
        pixel_smoothing : float
            How many pixels we should consider when smoothing.
        min_num_pts_per_pixel : float, optional
            Minimum number of points per pixel (after smoothing).
        datasetids : np.ndarray, optional
            Array of dataset ids. Must be of shape (N).
            If provided, features are independently computed for each unique dataset id.

        """
        self._features_extracted = False
        self._xy = None
        self._unique_labels = None
        self._datasetids = None
        self._results = {}
        self._labels = None
        self._colors = None
        self._num_points = None
        self.cluster_centers = None
        self._is_clustered = False
        self.inertia = None
        self._extract_features(
            xy, labels, pixel_width, pixel_smoothing, min_num_pts_per_pixel, datasetids
        )

    def fit(
        self,
        num_clusters: int,
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        """
        Fit the clustering model on the extracted features.

        Parameters
        ----------
        num_clusters : int
            Number of clusters to form.
        seed : int
            Seed for random initialization.
        kmeans_kwargs : dict, optional
            Dictionary with keyword arguments to be passed to
            scikit-learn's MiniBatchKMeans constructor.
        """

        self._cluster(num_clusters, seed, kmeans_kwargs)
        return self

    def predict(
        self,
        output: Literal["marker", "pixel", "anndata", "geojson", "colors", "connected"],
        adata_cluster_key: str = "Clusters",
        grow: Optional[int] = None,
        min_area: int = 1,
    ) -> Any:
        """
        Fit and predict the output based on the specified format after fitting the clustering model.

        Parameters
        ----------
        num_clusters : int
            Number of clusters to form.

        output : Literal['marker', 'pixel', 'anndata','geojson', 'colors'] specifying the output format, where

            * marker : np.ndarray
                - Returns the cluster label for each marker as an ndarray. Clusters equal to -1 correspond to background.

            * pixel : Tuple(np.ndarray, np.ndarray)
                - Returns a label mask of size `height` times `width` where each pixel is labelled by a cluster.
                - Also returns the parameters, `T`, such that input markers, `xy`, are mapped onto the label mask via `T[0] * xy + T[1]`.
                - The optional parameter `grow` can be set to an integer value for growing the label mask's foreground pixels.

            * anndata : AnnData, returns an AnnData object with:
                - Marker-count vectors stored in `adata.X`.
                - Clusters stored in `adata.obs[adata_cluster_key]`.
                - Position of each pixel stored in `adata.obsm["spatial"]`.
                - Marker position and cluster per marker stored in `adata.uns`.

                Requires that the package `anndata` is installed.

            * geojson : Dict | List:
                - Returns a dictionary or list with geojson polygons

            * colors : np.ndarray
                - An ndarray of colors (hex) for each cluster. Similar clusters will have similar colors. Last entry correspond to the background color.

            * connected : Tuple[np.ndarray, int, np.ndarray, np.ndarray], returns connected components in the label mask.
                Output comes as a tuple with four values:
                    - `connected` is a label mask where each connected component is uniquely labelled.
                    - `num_components` is an integer indicating the number of unique connected components.
                    - `label_mask` is the label mask
                    - `tform` contains the slope and instersect so that input markers' positions, `xy`, can be mapped onto the label mask via `tform[0] * xy + tform[1]`
                The optional parameter, `grow`, can be used to grow the size of the label mask.
                The optional parameter, `min_area`, can be used to remove small connected components in the geojson polygons
                    or in the connected component label mask.

        seed : int, optional
            Random seed for clustering.

        adata_cluster_key : str, optional
            Key indicating which column to store the cluster ids in an AnnData object (default: 'Clusters').

        grow : Optional[int], optional
            If provided, the number of pixels to grow the foreground regions in the label mask.

        Returns
        -------
        Points2Regions
            Updated instance of the Points2Regions class.

        """

        if not self._is_clustered:
            raise ValueError(
                "Must run the method `.fit(...)` before `.predict(...)`, or use `.fit_predict(...)`"
            )
        if output == "marker":
            return self._get_clusters_per_marker()
        elif output == "pixel":
            return self._get_labelmask(grow=grow)
        elif output == "anndata":
            return self._get_anndata(cluster_key_added=adata_cluster_key)
        elif output == "geojson":
            return self._get_geojson(grow=grow, min_area=min_area)
        elif output == "colors":
            return self._get_cluster_colors(hex=True)
        elif output == "connected":
            label_mask_args = self._get_labelmask(grow=grow)
            return self._get_connected_components(label_mask_args, min_area=min_area)
        else:
            valid_inputs = {
                "marker",
                "pixel",
                "anndata",
                "geojson",
                "colors",
                "connected",
            }
            raise ValueError(
                f"Invalid value for `output` {output}. Must be one of the following: {valid_inputs}."
            )

    def fit_predict(
        self,
        num_clusters: int,
        output: Literal["marker", "pixel", "anndata", "geojson", "colors", "connected"],
        seed: int = 42,
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
        adata_cluster_key: str = "Clusters",
        grow: Optional[int] = None,
        min_area: int = 1,
    ) -> Any:
        """
        Fit and predict the output based on the specified format after fitting the clustering model.

        Parameters
        ----------
        num_clusters : int
            Number of clusters to form.

        output : Literal['marker', 'pixel', 'anndata','geojson', 'colors'] specifying the output format, where

            * marker : np.ndarray
                - Returns the cluster label for each marker as an ndarray. Clusters equal to -1 correspond to background.

            * pixel : Tuple(np.ndarray, np.ndarray)
                - Returns a label mask of size `height` times `width` where each pixel is labelled by a cluster.
                - Also returns the parameters, `T`, such that input markers, `xy`, are mapped onto the label mask via `T[0] * xy + T[1]`.
                - The optional parameter `grow` can be set to an integer value for growing the label mask's foreground pixels.

            * anndata : AnnData, returns an AnnData object with:
                - Marker-count vectors stored in `adata.X`.
                - Clusters stored in `adata.obs[adata_cluster_key]`.
                - Position of each pixel stored in `adata.obsm["spatial"]`.
                - Marker position and cluster per marker stored in `adata.uns`.

                Requires that the package `anndata` is installed.

            * geojson : Dict | List:
                - Returns a dictionary or list with geojson polygons

            * colors : np.ndarray
                - An ndarray of colors (hex) for each cluster. Similar clusters will have similar colors. Last entry correspond to the background color.

            * connected : Tuple[np.ndarray, int, np.ndarray, np.ndarray], returns connected components in the label mask.
                Output comes as a tuple with four values:
                    - `connected` is a label mask where each connected component is uniquely labelled.
                    - `num_components` is an integer indicating the number of unique connected components.
                    - `label_mask` is the label mask
                    - `tform` contains the slope and instersect so that input markers' positions, `xy`, can be mapped onto the label mask via `tform[0] * xy + tform[1]`
                The optional parameter, `grow`, can be used to grow the size of the label mask.
                The optional parameter, `min_area`, can be used to remove small connected components in the geojson polygons
                    or in the connected component label mask.

        seed : int, optional
            Random seed for clustering.

        kmeans_kwargs : dict, optional
            Dictionary with keyword arguments to be passed to
            scikit-learn's MiniBatchKMeans constructor.

        adata_cluster_key : str, optional
            Key indicating which column to store the cluster ids in an AnnData object (default: 'Clusters').

        grow : Optional[int], optional
            If provided, the number of pixels to grow the foreground regions in the label mask.

        Returns
        -------
        Points2Regions
            Updated instance of the Points2Regions class.

        """

        self.fit(num_clusters, kmeans_kwargs, seed)
        return self.predict(output, adata_cluster_key, grow, min_area)

    def _extract_features(
        self,
        xy: np.ndarray,
        labels: np.ndarray,
        pixel_width: float,
        pixel_smoothing: float,
        min_num_pts_per_pixel: float = 0.0,
        datasetids: Optional[np.ndarray] = None,
    ):
        """
        Extracts features from input data.

        Parameters
        ----------
        xy : np.ndarray
            Array of coordinates. Must be of shape (N x 2).
        labels : np.ndarray
            Array of labels. Must be of shape (N).
        pixel_width : float
            Width of a pixel in the rasterized image.
        pixel_smoothing : float
            How many pixels we should consider when smoothing.
        min_num_pts_per_pixel : float, optional
            Minimum number of points per pixel (after smoothing).
        datasetids : np.ndarray, optional
            Array of dataset ids. Must be of shape (N).
            If provided, features are independently computed for each unique dataset id.

        Returns
        -------
        Points2Regions
            Updated instance.

        Raises
        ------
        ValueError
            If input shapes are incompatible.
        """

        # Set clusters
        self._xy = np.array(xy, dtype="float32")

        # Set labels
        self._labels = labels
        self._unique_labels = np.unique(labels)
        # Set dataset ids
        self._datasetids = datasetids
        # Create list for slicing data by dataset id
        if self._datasetids is not None:
            unique_datasetids = np.unique(self._datasetids)
            iterdata = [
                (
                    data_id,
                    (
                        self._xy[self._datasetids == data_id, :],
                        self._labels[self._datasetids == data_id],
                    ),
                )
                for data_id in unique_datasetids
            ]
        else:
            iterdata = [("id", (self._xy, self._labels))]

        # Get features per dataset
        # Store in complicated dicionary
        self._results = {}
        for datasetid, (xy_slice, labels_slice) in iterdata:
            self._results[datasetid] = inverse_distance_interpolation(
                xy_slice,
                labels_slice,
                self._unique_labels,
                pixel_width,
                pixel_smoothing,
                min_num_pts_per_pixel,
            )

            self._features_extracted = True
        return self

    def _cluster(
        self,
        num_clusters: int,
        seed: int = 42,
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Performs clustering on the extracted features.
        The method `extract_feature` must be called
        before calling `cluster`.

        The results can be extracted using the methods:
            - `get_clusters_per_marker` to get cluster per marker
            - `get_clusters_per_pixel` to get cluster for each pixel
            - `get_label_mask` to get the clusters as a label mask
            - `get_rgb_image` to get an RGB image where similar
                colors indicate similar clusters.
            - `get_geojson` to get the clusters as geojson polygons
            - `get_anndata` to the clusters in an anndata object
            - `get_cluster_colors` to get colors for each cluster.
                Similar clusters will be colored similarly.

        Parameters
        ----------
        num_clusters : int
            Number of clusters.
        seed : int, optional
            Random seed.
        kmeans_kwargs : dict, optional
            Dictionary with keyword arguments to be passed to
            scikit-learn's MiniBatchKMeans constructor.

        Returns
        -------
        Points2Regions
            Updated instance.
        """

        # Create train features
        self.X_train = sp.vstack(
            [
                result["features"][result["passed_threshold"]]
                for result in self._results.values()
            ]
        )

        default_kmeans_kwargs = dict(
            init="k-means++",
            max_iter=100,
            batch_size=1024,
            verbose=0,
            compute_labels=True,
            random_state=seed,
            tol=0.0,
            max_no_improvement=10,
            init_size=None,
            n_init="auto",
            reassignment_ratio=0.005,
        )

        if kmeans_kwargs is not None:
            for key, val in kmeans_kwargs.items():
                if key in default_kmeans_kwargs:
                    default_kmeans_kwargs[key] = val

        if isinstance(default_kmeans_kwargs["init"], sp.spmatrix):
            n_kmeans_clusters = default_kmeans_kwargs["init"].shape[0]
            use_hierarchial = False
        elif isinstance(default_kmeans_kwargs["init"], np.ndarray):
            n_kmeans_clusters = default_kmeans_kwargs["init"].shape[0]
            use_hierarchial = False
        else:
            n_kmeans_clusters = int(1.5 * num_clusters)
            use_hierarchial = True

        kmeans = KMeans(n_kmeans_clusters, **default_kmeans_kwargs)

        kmeans = kmeans.fit(self.X_train)
        self.inertia = kmeans.inertia_

        # Merge clusters using agglomerative clustering
        if use_hierarchial:
            clusters = _merge_clusters(kmeans, num_clusters)
            # Compute new cluster centers
            _, self.cluster_centers = _compute_cluster_center(
                clusters, kmeans.cluster_centers_
            )
        else:
            clusters = kmeans.labels_
            self.cluster_centers = kmeans.cluster_centers_

        # Iterate over datasets
        for datasetid, result_dict in self._results.items():
            # Get features and boolean indices for features passing threshold
            features, passed_threshold, pix2marker_ind = (
                result_dict["features"],
                result_dict["passed_threshold"],
                result_dict["pix2marker_ind"],
            )

            # Get kmeans clusters for each dataset id
            kmeans_clusters = kmeans.predict(features[passed_threshold])

            # Get clusters per pixel
            merged_clusters = clusters[kmeans_clusters]
            cluster_per_pixel = np.zeros(features.shape[0], dtype="int") - 1
            cluster_per_pixel[passed_threshold] = merged_clusters

            # Get clusters per marker
            cluster_per_marker = cluster_per_pixel[pix2marker_ind]

            # Store result in a dictioanry
            self._results[datasetid]["cluster_per_marker"] = cluster_per_marker
            self._results[datasetid]["cluster_per_pixel"] = cluster_per_pixel

        self._num_points = len(self._xy)

        # Compute colors
        self._set_cluster_colors()

        self._is_clustered = True

        return self

    def _get_cluster_colors(self, hex: bool = False) -> np.ndarray:
        """
        Retrieves cluster colors.

        Parameters
        ----------
        hex : bool, optional
            Flag indicating whether to return colors in hexadecimal format.

        Returns
        -------
        np.ndarray
            If hex is False, returns an ndarray of cluster colors.
            If hex is True, returns an ndarray of cluster colors in hexadecimal format.
        """
        if not hex:
            return np.array(self._colors)
        else:
            return np.array([_rgb_to_hex(rgb) for rgb in self._colors])

    def _set_cluster_colors(self):
        # If only one cluster, choose green
        if len(self.cluster_centers) == 1:
            self._colors = np.array([0, 1.0, 0]).reshape((1, -1))
            return

        # Compute distances between clusters
        D = pairwise_distances(self.cluster_centers)

        # Map each factor to a color
        embedding = TSNE(
            n_components=2,
            perplexity=min(len(self.cluster_centers) - 1, 30),
            init="random",
            metric="precomputed",
            random_state=1,
        ).fit_transform(D)

        # We interpret the 2D T-SNE points as points in a CIELAB space.
        # We then convert the CIELAB poitns to RGB.
        mu = 5
        out_ma = 128
        out_mi = -128
        mi, ma = np.percentile(embedding, q=mu, axis=0, keepdims=True), np.percentile(
            embedding, q=100 - mu, axis=0, keepdims=True
        )
        colors = np.clip((embedding - mi) / (ma - mi), 0.0, 1.0)
        colors = (out_ma - out_mi) * colors + out_mi
        colors = np.hstack((np.ones((len(colors), 1)) * 70, colors))
        self._colors = lab2rgb(colors)
        self._colors = np.vstack((self._colors, np.zeros(3)))

    def _get_labelmask(self, grow: Optional[float] = None) -> np.ndarray:
        """
        Generates label mask.

        Parameters
        ----------
        grow : float, optional
            Fill background pixels by growing foreground regions by a `grow` amount of pixels.

        Returns
        -------
        Tuple[np.ndarray, Tuple[float, float]]
            Label mask as an ndarray and transformation coefficients `T`,
            such that `xy * T[0] + T[1]` transforms the location of an
            input marker, `xy`, to the correct pixel in the label mask.
        """
        masks = {}
        for datasetid, result in self._results.items():
            grid_props = result["grid_props"]

            # Create label mask
            clusters = result["cluster_per_pixel"]
            label_mask = np.zeros(grid_props["grid_size"], dtype="int")
            label_mask[tuple(ind for ind in grid_props["grid_coords"])] = clusters + 1

            # Upscale the mask to match data
            scale = grid_props["grid_scale"]
            shift = grid_props["grid_offset"]
            T = (scale, -shift * scale)
            masks[datasetid] = (label_mask.T - 1, T)

        if grow is not None:
            for datasetid, (mask, T) in masks.items():
                # Mask foreground from background
                binary_mask = mask != -1

                # Compute distance from each background pixel to foreground
                distances = edt(~binary_mask)

                # Get coordinates of background pixels that are close
                # to foreground
                yx_bg = np.vstack(np.where(distances < grow)).T
                yx_fg = np.vstack(np.where(binary_mask)).T
                _, ind = KDTree(yx_fg).query(yx_bg, k=1)
                ind = np.array(ind).flatten()

                mask[yx_bg[:, 0], yx_bg[:, 1]] = mask[yx_fg[ind, 0], yx_fg[ind, 1]]

                # Erode to remove over-bluring near borders
                binary_mask = mask != -1
                binary_mask = binary_erosion(
                    binary_mask, iterations=int(grow), border_value=1
                )
                mask[~binary_mask] = -1
                masks[datasetid] = (mask, T)

        if len(self._results.keys()) == 1:
            return masks[datasetid]
        return masks

    def _get_anndata(self, cluster_key_added: str = "Clusters") -> Any:
        """
        Creates an AnnData object.

        Parameters
        ----------
        cluster_key_added : str
            Key indicating which column to store the cluster ids in (default: `Clusters`)

        Returns
        -------
        Any
            AnnData object with:
                - Marker-count vectors stored in `adata.X`.
                - Clusters stored in `adata.X`.
                - Position of each pixel stored in `adata.obsm["spatial"]`.
                - Marker position and cluster per marker stored in `adata.uns`.
        """
        # Create an adata object
        import anndata
        import pandas as pd

        print("Creating anndata")
        # Get position of bins for each group (library id)
        xy_pixel = np.vstack(
            [r["xy_pixel"][r["passed_threshold"]] for r in self._results.values()]
        )

        # Get labels of bins for each group (library id)
        labels_pixel = np.hstack(
            [
                r["cluster_per_pixel"][r["passed_threshold"]]
                for r in self._results.values()
            ]
        ).astype(str)

        obs = {}
        obs[cluster_key_added] = labels_pixel
        if len(self._results) > 1:
            obs["datasetid"] = np.hstack(
                [
                    [id] * len(r["cluster_per_pixel"][r["passed_threshold"]])
                    for id, r in self._results.items()
                ]
            )

        # Multiply back features with the norm
        norms = 1.0 / np.hstack(
            [r["norms"][r["passed_threshold"]] for r in self._results.values()]
        )
        norms[np.isinf(norms)] = 0
        norms = norms.reshape((-1, 1))

        # Remove the normalization
        X = self.X_train.multiply(norms).tocsc()

        # Remove the log transofrm.
        # X should be counts
        X.data = np.expm1(X.data)

        # Create the anndata object
        adata = anndata.AnnData(X=X, obs=obs, obsm=dict(spatial=xy_pixel))

        adata.var_names = self._unique_labels
        adata.obs["datasetid"] = adata.obs["datasetid"].astype("int")

        adata.obs[cluster_key_added] = adata.obs[cluster_key_added].astype("category")

        if len(self._results) > 1:
            adata.obs["datasetid"] = adata.obs["datasetid"].astype("category")

        marker2pix_ind = []
        offset = 0
        for r in self._results.values():
            # Get indices of non empty bins
            non_empty = np.where(r["passed_threshold"])[0]

            # Remap each point in the dataset
            remap = {i: -1 for i in range(len(r["cluster_per_marker"]))}
            for new, old in enumerate(non_empty):
                remap[old] = new + offset
            marker2pix_ind.append([remap[i] for i in r["pix2marker_ind"]])
            offset += len(non_empty)

        marker2pix_ind = np.hstack(marker2pix_ind)

        # Create reads dataframe
        reads = {}
        reads["x"] = self._xy[:, 0]
        reads["y"] = self._xy[:, 1]
        reads["labels"] = self._labels
        reads[cluster_key_added] = self._get_clusters_per_marker()
        reads["pixel_ind"] = marker2pix_ind

        if self._datasetids is not None:
            reads["datasetid"] = self._datasetids
            reads["datasetid"] = reads["datasetid"].astype("int")

        # Create the dataframe
        reads = pd.DataFrame(reads)

        # Change the datatypes
        reads["labels"] = reads["labels"].astype("category")
        reads[cluster_key_added] = reads[cluster_key_added].astype("category")
        if self._datasetids is not None:
            reads["datasetid"] = reads["datasetid"].astype("category")

        # Add to anndata
        adata.uns["reads"] = reads
        return adata

    def _get_geojson(self, grow: int = None, min_area: int = 0) -> Union[Dict, List]:
        """
        Generates GeoJSON representation of the regions.



        Parameters
        ----------
        gorw : int
            The optional parameter, `grow`, can be used to grow the size of the label mask
            with this many pixels. Default 0.

        Returns
        -------
        Union[Dict, List]
            GeoJSON data.
        """

        # Get label mask and transformations
        if len(self._results) == 1:
            label_mask, tform = self._get_labelmask(grow=grow)
            geojson = labelmask2geojson(
                label_mask,
                scale=1.0 / tform[0],
                offset=-tform[1] / tform[0],
                min_area=min_area,
            )
        else:
            geojson = {}
            for datasetid, (label_mask, tform) in self._get_labelmask(grow=grow):
                geojson[datasetid] = labelmask2geojson(
                    label_mask,
                    scale=1.0 / tform[0],
                    offset=-tform[1] / tform[0],
                    min_area=min_area,
                )

        return geojson

    def _get_clusters_per_marker(self) -> np.ndarray:
        """
        Retrieves clusters per marker.

        Returns
        -------
        np.ndarray
            Clusters per marker.
        """
        cluster_per_marker = np.zeros(self._num_points, dtype="int")
        for datasetid, result in self._results.items():
            cluster_per_marker[self._get_slice(datasetid)] = result[
                "cluster_per_marker"
            ]

        return cluster_per_marker.copy()

    def _get_slice(self, datasetid):
        if self._datasetids is not None:
            return self._datasetids == datasetid
        else:
            return np.ones(len(self._xy), dtype="bool")

    def _get_connected_components(
        self,
        label_mask_result: Union[
            Dict[str, Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]
        ],
        min_area: int = 1,
    ) -> Tuple[np.ndarray, int]:
        """
        Get connected components in the label mask.
        This can be slow for high-resolution masks,

        Parameters
        ----------
        label_mask : np.ndarray
            Label mask.
        min_area : int, optional
            Minimum area for connected components.

        Returns
        -------
        Tuple[np.ndarray, int, np.ndarray, Tuple[float,float]]
            Labels of connected components, the number of components, the label mask
            and a tuple of transformation parameters that can be used to map each
            observed point onto the masks via `tform[0]*xy+tform[1]`
        """

        if not isinstance(label_mask_result, dict):
            dataset_dictionary = {"datasetid": label_mask_result}
        else:
            dataset_dictionary = label_mask_result

        output = {}

        for datasetid, result in dataset_dictionary.items():
            # Get the shape of the label image
            label_mask = result[0]
            tform = result[1]

            # Shift label mask so that 0 is background instead of -1
            label_mask = label_mask + 1

            N, M = label_mask.shape
            total_pixels = N * M

            # Create 1D arrays for row and column indices of the adjacency matrix
            row_indices = np.arange(total_pixels).reshape(N, M)
            col_indices = np.copy(row_indices)

            # Mask for pixels with the same label in horizontal direction
            mask_same_label_horizontal = (label_mask[:, :-1] == label_mask[:, 1:]) & (
                label_mask[:, :-1] != 0
            )

            # Include connections between pixels with the same label
            row_indices_horizontal = row_indices[:, :-1][
                mask_same_label_horizontal
            ].flatten()
            col_indices_horizontal = row_indices[:, 1:][
                mask_same_label_horizontal
            ].flatten()

            # Mask for pixels with the same label in vertical direction
            mask_same_label_vertical = (label_mask[:-1, :] == label_mask[1:, :]) & (
                label_mask[:-1, :] != 0
            )

            # Include connections between pixels with the same label
            row_indices_vertical = col_indices[:-1, :][
                mask_same_label_vertical
            ].flatten()
            col_indices_vertical = col_indices[1:, :][
                mask_same_label_vertical
            ].flatten()

            # Combine the horizontal and vertical connections
            r = np.concatenate([row_indices_horizontal, row_indices_vertical])
            c = np.concatenate([col_indices_horizontal, col_indices_vertical])

            # Create COO format data for the sparse matrix
            data = np.ones_like(r)

            # Create the sparse matrix using COO format
            graph_matrix = sp.coo_matrix(
                (data, (r, c)), shape=(total_pixels, total_pixels)
            )

            # Remove duplicate entries in the COO format
            graph_matrix = sp.coo_matrix(graph_matrix)

            # Run connected component labelling
            num_components, labels = sp.csgraph.connected_components(
                graph_matrix, directed=False
            )

            # Compute frequency of each connected component
            counts = np.bincount(labels, minlength=num_components)

            # Mask out small components
            mask = counts <= min_area

            # Relabel labels such that
            # small components have label 0
            # other labels have labels 1->N+1
            counts[mask] = 0

            num_large_components = np.sum(~mask)
            counts[~mask] = np.arange(1, num_large_components + 1)
            labels = counts[labels]

            # Shift so that labels are in [0, num_large_components)
            labels = labels - 1

            # Reshape to a grid
            labels = labels.reshape(label_mask.shape)

            output[datasetid] = (labels, num_large_components, label_mask, tform)

        if len(output) == 1:
            return output[datasetid]
        else:
            return output


from flask import abort, make_response
import os
import tempfile
import json
import hashlib

import h5py
import numpy as np
import pandas as pd


class Plugin:
    def __init__(self, app):
        self.app = app

    def getPythonCode(self):
        import inspect

        resp = make_response(
            inspect.getsource(inspect.getmodule(inspect.currentframe()))
        )
        return resp

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

        print(jsonParam)
        nclusters = int(jsonParam["nclusters"])
        min_pts_per_pixel = float(jsonParam["min_pts_per_pixel"])
        pixel_size = float(jsonParam["pixel_size"])
        pixel_smoothing = float(jsonParam["pixel_smoothing"])
        region_name = str(jsonParam["region_name"])
        seed = int(jsonParam["seed"])
        format = jsonParam["format"]

        if format == "GeoJSON polygons":
            mdl = Points2RegionClass(
                xy, labels, pixel_size, pixel_smoothing, min_pts_per_pixel
            )
            c = mdl.fit_predict(nclusters, output="geojson", seed=seed)
            print(c)

            strOutput = json.dumps(c)

        else:
            mdl = Points2RegionClass(
                xy, labels, pixel_size, pixel_smoothing, min_pts_per_pixel
            )
            c = mdl.fit_predict(nclusters, output="marker", seed=seed)
            strOutput = np.array2string(c, separator=",", threshold=c.shape[0])

        with open(cacheFile, "w") as f:
            f.write(strOutput)
        resp = make_response(strOutput)
        return resp
