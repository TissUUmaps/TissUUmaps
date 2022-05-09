from flask import abort, make_response
import logging
import os
import pandas as pd
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix
from sklearn.cluster import MiniBatchKMeans as KMeans
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops, approximate_polygon, find_contours
import hashlib, tempfile


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

    def Points2Regions(self, jsonParam):
        if not jsonParam:
            logging.error("No arguments, aborting.")
            abort(500)

        cacheFile = self.getCacheFile(jsonParam)
        if os.path.isfile(cacheFile):
            with open(cacheFile, "r") as f:
                return make_response(f.read())

        csvPath = os.path.abspath(os.path.join(self.app.basedir, jsonParam["csv_path"]))

        df = pd.read_csv(csvPath)
        xy = df[[jsonParam["xKey"], jsonParam["yKey"]]].to_numpy()
        labels = df[jsonParam["clusterKey"]].to_numpy()

        sigma = jsonParam["sigma"]
        stride = jsonParam["stride"]
        region_name = jsonParam["region_name"]
        nclusters = jsonParam["nclusters"]
        expression_threshold = jsonParam["expression_threshold"]
        normalize_order = jsonParam["normalize_order"]

        regions = points2geojson(
            xy,
            labels,
            sigma,
            stride,
            region_name,
            nclusters,
            normalize_order,
            expression_threshold,
        )
        strRegions = json.dumps(regions)

        with open(cacheFile, "w") as f:
            f.write(strRegions)

        resp = make_response(strRegions)
        return resp


# COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

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


def binary_mask_to_polygon(binary_mask, tolerance=0, offset=None, scale=None):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        if offset is not None:
            contour = contour + offset  # .ravel().tolist()
        if scale is not None:
            contour = contour * scale
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        polygons.append(contour.tolist())

    return polygons


def create_gene_signatures(
    gene_labels,
    gene_position_xy,
    spatial_scale,
    gene_labels_unqiue=None,
    bin_stride=None,
    bin_type="gaussian_per_point",
):

    """
    Computes gene expression signatures in different "empty" bins by inpainting neighbouring gene expression markers.
    The inpainting is done by:
        1.
            Each "empty" bin is connected to its neighbouring, "known", gene expression bins through edges.
            A known gene expression bin is a one-hot-encoded vector that labels a particular gene.
            Each edge has a weight that is given by exp(-0.5 d_ij^2 / sigma^2 ), where d_ij is the distance
            between the empty node i and the known node j.
        2. The gene expression of the the empty bin is obtained by a weighted sum of the neighbouring known bins.

    The empty bins can either be placed on a rectangular grid or ontop of each known node.

    Input:
        - gene_labels: The categorical label of each gene, (#genes long np.array).
        - gene_position_xy: Position of each gene (#genes x 2 size np.array)
        - Spatial scale, i.e., the sigma of a Gaussian kernel. Scalar
        - gene_labels_unique: unique list of labels. If not provided, it is automagically computed using np.unique(gene_labels)
        - bin_stride: Stride between bins. If not provided it is set to spatial scale.
        - bin_type: option for setting the type of bins. If set to gaussian_grid it will use bins on a grid. If set to gaussian_per_point
            it will compute bins centered around each point in gene_position_xy.
    Output:
        Gene vectors (#genes x #gene_types) gene profile vectors
        Coord       (#genes x 2) spatial coordinate of each vector.
    """

    ok_bin_types = ["gaussian_grid", "gaussian_per_point", "gaussian_grid_fast"]

    if bin_type not in ok_bin_types:
        raise ValueError(f"bin_type must be one of the following: {ok_bin_types}")

    if bin_stride is None:
        bin_stride = spatial_scale

    if gene_labels_unqiue is None:
        gene_labels_unqiue = np.unique(gene_labels)

    # Create unique numeric labels
    gene_labels_unique_map = {str(k): i for i, k in enumerate(gene_labels_unqiue)}
    gene_labels_numeric = np.array(
        [gene_labels_unique_map[str(k)] for k in gene_labels]
    )

    gene_labels_numeric_unqiue = np.array(
        [gene_labels_unique_map[str(label)] for label in gene_labels_unqiue]
    )
    n_genes = len(gene_labels_unqiue)
    n_pts = gene_position_xy.shape[0]
    if bin_type == "gaussian_grid":
        # Compute grid start and stop
        start = gene_position_xy.min(axis=0)
        stop = gene_position_xy.max(axis=0) + 0.5 * bin_stride
        # Compute location of each bin
        y, x = np.meshgrid(
            np.arange(start[1], stop[1], bin_stride),
            np.arange(start[0], stop[0], bin_stride),
        )
        bin_coords = np.array([x.ravel(), y.ravel()]).T
        # Compute gene vectors
        vectors = np.eye(n_genes)
        p = cKDTree(gene_position_xy).query_ball_point(
            bin_coords, spatial_scale * 3, p=2
        )
        q = [np.ones(len(l)) * i for i, l in enumerate(p) if len(l) > 0]
        p = np.concatenate(p).astype("int")
        q = np.concatenate(q).astype("int")
        # Affinities
        aff = np.sum((gene_position_xy[p, :] - bin_coords[q, :]) ** 2, axis=1)
        aff = np.exp(-0.5 * aff / (spatial_scale**2))
        # Sum neighbours
        gene_each_point_one_hot = np.array(
            [vectors[:, gene_labels_numeric[i]] for i in range(n_pts)]
        )
        gene_vectors = (
            csc_matrix((aff, (q, p)), shape=(bin_coords.shape[0], n_pts))
            @ gene_each_point_one_hot
        )
    elif bin_type == "gaussian_grid_fast":
        # Bin each marker onto a 2D grid
        m = gene_position_xy.min(axis=0)
        k = bin_stride
        xy_downsamples = np.array((gene_position_xy - m) // k, dtype="int")
        dim = np.append(xy_downsamples.max(axis=0) + 1, n_genes)
        linearind = np.ravel_multi_index(
            (xy_downsamples[:, 0], xy_downsamples[:, 1], gene_labels_numeric), dim
        )
        linearind_unique, counts = np.unique(linearind, return_counts=True)
        x, y, c = np.unravel_index(linearind_unique, dim)
        gene_vectors = np.zeros(dim)
        gene_vectors[x, y, c] = counts
        alpha = spatial_scale / bin_stride * 2 * np.pi
        for l in gene_labels_numeric_unqiue:
            gene_vectors[:, :, l] = alpha * gaussian_filter(
                gene_vectors[:, :, l], spatial_scale / bin_stride
            )
        gene_vectors = gene_vectors.reshape(
            (-1, len(gene_labels_numeric_unqiue)), order="F"
        )
        x, y = np.meshgrid(np.arange(dim[0]), np.arange(dim[1]))
        bin_coords = k * (np.array([x.ravel(), y.ravel()]).T) + m
    elif bin_type == "gaussian_per_point":
        bin_coords = gene_position_xy.copy()
        vectors = np.eye(n_genes)
        p = cKDTree(gene_position_xy).query_ball_point(
            bin_coords, spatial_scale * 3, p=2
        )
        q = [np.ones(len(l)) * i for i, l in enumerate(p) if len(l) > 0]
        p = np.concatenate(p).astype("int")
        q = np.concatenate(q).astype("int")
        aff = np.sum((gene_position_xy[p, :] - bin_coords[q, :]) ** 2, axis=1)
        aff = np.exp(-0.5 * aff / (spatial_scale**2))
        gene_each_point_one_hot = np.array(
            [vectors[:, gene_labels_numeric[i]] for i in range(n_pts)]
        )
        gene_vectors = (
            csc_matrix((aff, (q, p)), shape=(bin_coords.shape[0], n_pts))
            @ gene_each_point_one_hot
        )
    return gene_vectors, bin_coords, gene_labels_numeric_unqiue


def preprocess_vectors(
    gene_vectors,
    gene_vector_coords,
    threshold=None,
    logtform=True,
    normalize=True,
    ord=1,
):
    """
    The parameter ord is the order of the normalization
    and must be one of the following:

    inf    max(abs(x))
    0      sum(x != 0)
    1      sum(abs(x))
    2      sqrt(sum(x^2))
    """
    if threshold is not None:
        ind = gene_vectors.sum(axis=1) > threshold
        gene_vectors = gene_vectors[ind, :]
        gene_vector_coords = gene_vector_coords[ind, :]

    if logtform:
        gene_vectors = np.log(gene_vectors + 1)
    if normalize:
        gene_vectors = gene_vectors / np.linalg.norm(
            gene_vectors, ord=ord, axis=1, keepdims=True
        )
    return gene_vectors, gene_vector_coords


def cluster_signatures(
    gene_vectors, n_clusters=6, threshold=None, logtform=True, normalize=True, ord=1
):
    """
    Cluster gene vectors (#ngenes x #gene types).
    """
    kmeans = KMeans(n_clusters, random_state=42).fit(gene_vectors)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    return labels, cluster_centers


def makejson(
    regionname: str,
    labels_numeric_unique,
    labels: np.array,
    xy: np.array,
    bin_stride,
    colors,
):
    xy_lowres = (xy.astype("float32") // bin_stride).astype("int")
    sz = xy_lowres.max(axis=0) + 1
    labelmask = np.zeros((sz[0], sz[1]), dtype="uint8")
    for label in labels_numeric_unique:
        ind = labels == label
        labelmask[xy_lowres[ind, 0], xy_lowres[ind, 1]] = label + 1

    polygons = []
    cluster_names = [f"Region {l+1}" for l in labels_numeric_unique]
    for region in regionprops(labelmask):
        # take regions with large enough areas
        print(region.label)
        contours = binary_mask_to_polygon(
            region.image,
            offset=np.array([region.bbox[0] + 0.5, region.bbox[1] + 0.5]),
            scale=bin_stride,
        )
        polygons.append(contours)

    json = polygons2json(polygons, regionname, cluster_names, colors=colors)
    return json


def points2geojson(
    xy: np.array,
    labels: np.array,
    sigma: float,
    stride: int,
    region_name: str = "My regions",
    nclusters: int = 4,
    normalize_order: int = 1,
    expression_threshold: float = 3,
):
    colors = [COLORS[k % len(COLORS)] for k in range(nclusters)]

    # Create neighbourhood vectors
    gene_vectors, gene_vector_coords, gene_labels_unique = create_gene_signatures(
        labels, xy, sigma, bin_stride=stride, bin_type="gaussian_grid_fast"
    )
    # Filter genes
    gene_vectors, gene_vector_coords = preprocess_vectors(
        gene_vectors,
        gene_vector_coords,
        threshold=expression_threshold,
        logtform=False,
        normalize=True,
        ord=normalize_order,
    )

    # Mini batch kmeans
    clusterlabels, centers = cluster_signatures(gene_vectors, nclusters)

    # Geojson
    json = makejson(
        region_name,
        gene_labels_unique,
        clusterlabels,
        gene_vector_coords,
        stride,
        colors,
    )

    return json


def test():
    import pandas as pd
    import json

    data = pd.read_csv("example_data.csv")
    xy = data[["x", "y"]].to_numpy()
    labels = data["name"].to_numpy()
    sigma = 40
    stride = int(0.5 * sigma)
    geojson = points2geojson(
        xy,
        labels,
        sigma,
        stride,
        region_name="My regions",
        nclusters=8,
        normalize_order=1,
        expression_threshold=3,
    )
    with open(f"regions.json", "w") as fp:
        json.dump(geojson, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    test()
