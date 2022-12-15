import hashlib
import json
import logging
import os
import tempfile

import numpy as np
import pandas as pd
from flask import abort, make_response
from scipy.sparse import csr_matrix, vstack
from skimage.measure import approximate_polygon, find_contours, regionprops
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize as sknormalize

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

        if scale is not None:
            contour = contour * scale
        if offset is not None:
            contour = contour + offset  # .ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        polygons.append(contour.tolist())

    return polygons


def labelmask2geojson(
    labelmask, region_name: str = "My regions", scale: float = 1.0, offset: float = 0
):
    nclusters = np.max(labelmask)
    colors = [COLORS[k % len(COLORS)] for k in range(nclusters)]

    # Make JSON
    polygons = []
    cluster_names = [f"Region {l+1}" for l in np.arange(1, nclusters + 1)]
    for region in regionprops(labelmask):
        # take regions with large enough areas
        contours = binary_mask_to_polygon(
            region.image,
            offset=scale * np.array(region.bbox[0:2]) + offset + 0.5 * scale,
            scale=scale,
        )
        polygons.append(contours)
    json = polygons2json(polygons, region_name, cluster_names, colors=colors)
    return json


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

        regions = FastCluster(
            xy, labels, stride, sigma, nclusters, min_density=expression_threshold
        ).get_geojson(region_name=region_name)
        strRegions = json.dumps(regions)

        with open(cacheFile, "w") as f:
            f.write(strRegions)

        resp = make_response(strRegions)
        return resp


class Rasterizer:
    def __init__(self, bin_width: float, xy: np.ndarray, labels: np.ndarray) -> None:
        self.bin_width = bin_width
        npts = len(labels)
        labels_unique = np.unique(labels)
        __label2num = {l: i for i, l in enumerate(labels_unique)}

        self.__labels_numeric = np.array(list(map(__label2num.get, labels)))
        self.__labels_numeric_unique = np.array(list(__label2num.values()))

        # Get id of each bin
        self.offset = np.min(xy, axis=0, keepdims=True)
        positional_ids = (xy - self.offset) // bin_width
        positional_ids = positional_ids.astype("int")

        grid_shape = tuple(np.max(positional_ids, axis=0).astype("int") + 1)

        positional_linear_ids = np.ravel_multi_index(positional_ids.T, dims=grid_shape)
        (
            self._unique_positions_linear,
            positional_linear_ids_zero_based,
        ) = self.__reindex(positional_linear_ids)
        self._unique_positions = np.array(
            np.unravel_index(self._unique_positions_linear, shape=grid_shape)
        ).T

        # Center of mass for each bin
        _, _, xy2bin = np.unique(
            positional_linear_ids_zero_based,
            return_index=True,
            return_inverse=True,
        )

        com_map = csr_matrix((np.ones(npts), (xy2bin, np.arange(npts))), dtype="bool")
        self._bin_com = (com_map @ xy) / (com_map @ np.ones((npts, 1)))

        bin_ids = np.vstack((positional_linear_ids_zero_based, self.__labels_numeric)).T

        u, counts = np.unique(bin_ids, axis=0, return_counts=True)

        g = u[:, 0]
        a = np.array(
            np.unravel_index(self._unique_positions_linear[g], shape=grid_shape)
        )

        masks = [u[:, 1] == label for label in self.__labels_numeric_unique]
        self._features = [
            csr_matrix((counts[mask], (a[0, mask], a[1, mask])), shape=grid_shape)
            for mask in masks
        ]

        m = len(self._unique_positions)
        self.grid_shape = grid_shape
        self.grid_coords = self._unique_positions
        self._pt2bin = positional_linear_ids_zero_based
        self._active_bins = csr_matrix(
            (
                np.repeat(1, m),
                (self._unique_positions[:, 0], self._unique_positions[:, 1]),
            ),
            shape=grid_shape,
        )

    def __reindex(self, indices):
        u, i = np.unique(indices, return_inverse=True)
        re = np.arange(len(u))
        return u, re[i]

    def threshold(self, min_density: float = 0):
        if min_density == 0:
            return

        s = np.zeros(self.grid_shape)
        for i, n in enumerate(self._features):
            s = s + n
        counts = s.ravel()
        if min_density == -1:
            logcounts = np.array(np.log(counts + 1)).ravel()
            is_bg = logcounts < 0.5
            gmm = GaussianMixture(n_components=2, covariance_type="diag")
            gmm.fit(logcounts[~is_bg].reshape((-1, 1)))
            bglabel = np.argmax(gmm.means_)
            fglabel = np.argmin(gmm.means_)

            # Predict background
            is_bg[~is_bg] = gmm.predict(logcounts[~is_bg].reshape((-1, 1))) == bglabel
            mask = csr_matrix(is_bg.reshape(self.grid_shape))
        else:
            mask = csr_matrix(s > min_density, dtype="float32")
        for i, n in enumerate(self._features):
            self._features[i] = n.multiply(mask)
        pass

    def diffuse(self, sigma: float = 1.0):
        if self.bin_width is not None:
            sigma = sigma / self.bin_width
        GX = csr_matrix(self.__make_gaussian(sigma, self.grid_shape[0]))
        GY = csr_matrix(self.__make_gaussian(sigma, self.grid_shape[1]).T)

        self._features = [GX @ X @ GY for X in self._features]

    def __make_gaussian(self, sigma, shape):
        mu = np.arange(shape).reshape((-1, 1))
        x = np.arange(shape)
        d2 = (mu - x) ** 2
        gaussian = np.exp(-d2 / (2 * sigma * sigma)) * (d2 < (9 * sigma * sigma))
        gaussian = gaussian / gaussian.max(axis=1, keepdims=True)
        return gaussian

    def which_bin(self):
        return self._pt2bin

    def bin_pos(self):
        return self._bin_com

    def extract(self, sparse: bool = True, normalize: bool = False):
        features = vstack(
            [f.reshape((1, np.prod(self.grid_shape))) for f in self._features]
        ).T.tocsr()
        if not sparse:
            features = np.array(features.todense())
        if normalize:
            features = sknormalize(features, norm="l1")
        return self._bin_com, features

    def extract_grid(self, sparse: bool = True, normalize: bool = False):
        features = (
            vstack(self._features, dtype="float32")
            .reshape((len(self._features), np.prod(self.grid_shape)))
            .T.tocsr()
        )
        features_query = features[self._unique_positions_linear, :]

        ind = np.arange(np.prod(self.grid_shape))
        mass = np.array(features.sum(axis=1)).ravel()
        keepind = mass > 0
        query_pass_threshold = np.array(features_query.sum(axis=1)).ravel() > 0
        features = features[keepind, :]
        ind = ind[keepind]
        xy = np.unravel_index(ind, self.grid_shape)
        if not sparse:
            features = np.array(features.todense())
            features_query = np.array(features_query.todense())
        if normalize:
            features = sknormalize(features, norm="l1")
            features_query = sknormalize(features_query, norm="l1")

        return xy, features, features_query, query_pass_threshold


class FastCluster:
    def __init__(
        self,
        xy: np.ndarray,
        labels: np.ndarray,
        bin_width: float,
        sigma: float,
        n_clusters: int,
        min_density: float = 0,
        random_state: int = 42,
    ) -> None:

        self.raster = Rasterizer(bin_width, xy, labels)
        self.raster.diffuse(sigma=sigma)
        self.raster.threshold(min_density)

        self.kmeans = KMeans(n_clusters, random_state=random_state)
        (
            self.grid_coords,
            self.features,
            self.features_query,
            pass_threshold,
        ) = self.raster.extract_grid(normalize=True)
        self.kmeans.fit(self.features_query[pass_threshold])
        self.labels_grid = self.kmeans.predict(self.features) + 1
        self.labels = self.kmeans.predict(self.features_query) + 1
        self.labels[~pass_threshold] = -1
        self.centroids = self.kmeans.cluster_centers_
        self.bin_width = bin_width

    def get_label_per_point(self):
        return self.labels[self.raster.which_bin()]

    def get_label_per_bin(self):
        return self.labels

    def get_position_per_bin(self):
        return self.raster._bin_com

    def get_geojson(self, region_name: str = "My regions"):
        labelmask = np.zeros(self.raster.grid_shape, dtype="int")
        labelmask[self.grid_coords[0], self.grid_coords[1]] = self.labels_grid
        labelmask = labelmask
        return labelmask2geojson(
            labelmask,
            scale=self.bin_width,
            offset=self.raster.offset,
            region_name=region_name,
        )


if __name__ == "__main__":
    from os.path import join

    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(join("data", "example_data", "data.csv"))
    xy = df[["x", "y"]].to_numpy()
    labels = df["gene"].to_numpy()

    c = FastCluster(xy, labels, bin_width=5, sigma=45, n_clusters=8, min_density=2)
    labels = c.get_label_per_point()
    import json

    d = c.get_geojson()
    with open(join("data", "example_data", "regions.json"), "w") as fp:
        json.dump(d, fp)
    df["semantic_label"] = labels
    df.to_csv(join("data", "example_data", "results.csv"))
