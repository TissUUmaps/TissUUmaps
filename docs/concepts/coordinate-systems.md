---
sidebar_position: 4
---

# Coordinate systems

## Comparison

| TissUUmaps        | [OpenSeadragon](https://openseadragon.github.io/examples/viewport-coordinates/) | [WebGL](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_model_view_projection)                               | Description                                                                               |
| ----------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Data/object       | Image                                                                           | Local/object                                                                                                                  | Raw data (i.e. pixel) coordinates                                                         |
| Layer             |                                                                                 |                                                                                                                               | Coordinate system shared among objects of different resolution, e.g. physical coordinates |
| World             | Viewport                                                                        | World                                                                                                                         | Global coordinate system in which the layers are positioned, e.g. after registration      |
| Viewport `[0, 1]` | Web `[0, canvas size]`                                                          | Normalized device coordinates (NDCs) `[-1, 1]`, equivalent to clip space when using orthographic projections for 2D rendering | The part that is currently visible in the browser (i.e., the HTML canvas element)         |
| Device            |                                                                                 |                                                                                                                               | Physical device (i.e. screen) pixels                                                      |

## Transformations

| Direction          | Transformation                                                                       |
| ------------------ | ------------------------------------------------------------------------------------ |
| Data -> Layer      | _layer coordinates_ = **_object-level similarity transform_** \* _data coordinates_  |
| Layer -> World     | _world coordinates_ = **_layer-level similarity transform_** \* _layer coordinates_  |
| World -> Viewport  | _viewport coordinates_ = (_world coordinates_ - _viewport origin_) / _viewport size_ |
| Viewport -> Device | _device coordinates_ = _device pixel ratio_ \* _viewport coordinates_                |

User-configurable variables are highlighted in bold.
