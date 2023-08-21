importScripts(
  "https://cdnjs.cloudflare.com/ajax/libs/Turf.js/6.5.0/turf.min.js"
);

self.onmessage = function (event) {
  const [region, offset] = event.data;
  offsetRegion(region, offset);
};

function offsetRegion(region, offset) {
  const multiPolygon = objectToArrayPoints(region.points);
  function createOffsetPolygon(multipolygon, offset) {
    const turfMultipolygon = turf.multiPolygon(multipolygon);
    const offsetPolygon = turf.buffer(turfMultipolygon, offset, {
      units: "kilometers",
    });
    return offsetPolygon.geometry;
  }
  try {
    const result = createOffsetPolygon(multiPolygon, offset);
    let points =
      result.type === "Polygon" ? [result.coordinates] : result.coordinates;
    self.postMessage(points);
  } catch {
    self.postMessage(null);
  }
}

function objectToArrayPoints(points) {
  return points.map((arr) =>
    arr.map((polygon) => polygon.map((point) => [point.x, point.y]))
  );
}
