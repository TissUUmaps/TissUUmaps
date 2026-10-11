---
sidebar_position: 6
---

# Rendering

**OpenSeadragon** is used for displaying images and labels. OpenSeadragon was chosen over other alternatives for maturity (stability, large community, active development) and legacy (TissUUmaps 3) reasons.

Custom **WebGL 2** shaders are used for rendering points and shapes. WebGL was chosen over WebGPU for browser compatibility (e.g. Firefox) and legacy (TissUUmaps 3) reasons.

## Images

Custom tile sources are used to enable additional file formats (e.g. TIFF, Zarr); the OME-Zarr tile source comes from the external `omezarr-tilesource` package.

Image data is either multi-channel (`getSizeC()` returns the number of channels), providing one tile source per channel, or not, providing a single grayscale or color tile source that is drawn as it is.

Multi-channel image data that implements `getTileData()` provides a single sample ("grayscale value") per pixel and channel. Each channel is then recolored by a data transfer (see [Code architecture](./code-architecture.md#tissuumapsrender)) that scales the values linearly between the channel's contrast limits, clamps them to `[0, 1]` and multiplies them with the channel color. Multi-channel image data that does not implement `getTileData()` is free to provide grayscale or RGB tile sources per channel, which are drawn as they are.

Colors and contrast limits are resolved by `ImageUtils.getChannelColor` and `ImageUtils.getChannelContrastLimits`, which the renderer and the channel settings panel share so that both show the same values. They are taken from the image's channel settings, falling back to those reported by the image data. Channels without a color are colorized with a default color derived from the channel index (`ImageUtils.getDefaultChannelColor`, based on QuPath's palette: red, green, blue, yellow, cyan, magenta, ...), except for the only channel of single-channel image data, which is colorized white; data providers should only report colors that their image metadata actually specifies and leave the choice of default colors to the renderer. Channels without contrast limits are stretched between quantile-based limits derived from the channel histogram (`ImageUtils.getDefaultContrastLimits`), if the image data provides one via `getChannelHistogram()`, and over the value range of their data type otherwise. Channel, image and layer opacity are applied by OpenSeadragon when drawing.

The image's channel view mode selects what is shown: `composite` blends every visible channel in its color, while `grayscale` and `color` show the image's active channel alone (`ImageUtils.getActiveChannel`), in white and in the channel color respectively. Which channels are shown is applied as opacity, so switching between composite and color, or changing the active channel, recolors no tile; entering or leaving grayscale recolors every channel, as it replaces the channel colors with white. The single-channel modes ignore channel visibility but keep channel opacity.

## Labels

Custom tile sources are used to enable loading of 8/16/32-bit signed or unsigned integer label masks (TIFF/Zarr).

Label tiles carry label IDs rather than colors. They are recolored by a data transfer that looks each ID up in a color table resolved per object from the label color, visibility and opacity configurations, with visibility and opacity folded into the alpha channel; ID `0` is transparent. A label image does not enumerate its labels (which would require scanning it), so the color table is resolved for the labels listed in the referenced table, if a configuration reads from it; every other label is resolved as it is first drawn - from the configurations that need no table (constant values, and random colors hashed from the ID), or with the default appearance - and memoized. Transfers are kept per tile source, so the navigator is recolored as well, and are only rebuilt when the data, a configuration or a group-to-value map a configuration references changes.

## Points

Points are rendered in multiple passes (i.e., one draw call per point cloud), each from its own vertex array and attribute buffers, in layer order and then object order. The vertex shader is executed once per point, the fragment shader once per fragment of each point sprite. Since point attributes (marker, size, color, visibility, opacity, ...) are individually configurable for each point, points are rendered naively (i.e., no instancing).

To enable partial attribute updates, data is loaded using separate buffers for each coordinate/property, so that a change to e.g. the color configuration of one point cloud only re-uploads the colors of that point cloud; likewise, a change to a group-to-value map only re-resolves the properties of the point clouds that reference it. However, point color, visibility and opacity values are packed into joint 32-bit RGBA values for memory efficiency. A property whose configuration is a constant has no buffer at all: its value is supplied as a generic vertex attribute per draw call, with the attribute array disabled, which saves the buffer, its upload and the resolution for the common case of constant markers, sizes and colors. Properties that apply to a point cloud as a whole are passed as per-pass uniforms rather than baked into the buffers: the data → world transform, the point size factor (object and layer point size factors, multiplied with the transform scales that the unit of the size configuration is subject to), and the layer and object opacity and visibility. Changing any of them, or reordering point clouds, therefore costs nothing but a redraw, and adding or removing a point cloud never touches the point-level data of the others. Marker shapes are loaded from a "marker atlas" texture, which holds signed distance fields (SDFs) that have been pre-multiplied by a constant factor and quantized into unsigned 8-bit integers for memory efficiency. The atlas is a 4×4 grid with one marker per RGBA channel, i.e. at most 64 markers.

## Shapes

Shapes are rendered in multiple passes (i.e., one draw call per shape cloud) using separate data textures for each shape cloud. A "compute shader approach" is employed, where the vertex shader merely runs on the four corners of a quad covering the shape cloud's bounds (dilated by how far the anti-aliased strokes reach beyond them) within the viewport, and the fragment shader implements a custom rendering pipeline (i.e., executed for all fragments of that quad). Shape clouds outside the viewport are skipped entirely.

Partial updates are enabled implicitly by using separate data textures for each shape cloud (update individual shape clouds) and each property (update individual shape cloud properties), including changes to the group-to-value maps a property references. However, shape fill/stroke color, visibility and opacity values are packed into joint 32-bit RGBA values for memory efficiency. As for points, the world → data transform and the layer and object opacity and visibility are per-pass uniforms rather than baked into the textures, so changing them never rebuilds a texture.

The custom rendering pipeline is based on scanline rendering, with the following modifications/optimizations:

- Scanline data (edge lists) are stored in separate data textures for each shape cloud
- Scanlines relate to the shape cloud bounds (as opposed to viewport/world bounds) to allow for infinite worlds
- Each scanline is divided into equally wide x-bins, each of which lists the shapes reaching into it, in the order they are composited in
- Shapes and edges are padded by a fraction (a render option) of the median shape height above and below, and shapes by the same fraction of the median shape width to the left and right, so that strokes reaching beyond a shape are still drawn, however large the scanlines and bins; strokes are therefore clipped where half the stroke width exceeds the padding
- For each scanline, edges are processed separately for each shape to ensure proper compositing
- Each shape holds a one-dimensional bounding box per scanline for rapidly skipping shapes
- An optimized winding number algorithm is used for point-in-polygon testing, with the even-odd rule (odd winding numbers are inside), so that holes are cut out whatever the orientation of their rings, which e.g. GeoJSON recommends but does not require
- An optimized point-to-segment distance algorithm is used for stroke drawing

Specifically, the approach works as follows:

- Construct scanline data for each shape cloud on the CPU and transfer them into data textures on the GPU
- For each shape cloud, call `gl.drawArrays()` with `gl.TRIANGLE_STRIP` for the quad covering its bounds within the viewport
  - The vertex shader will run once per quad corner, converting viewport coordinates to data coordinates
  - The fragment shader will run once for each fragment in the quad:
    1. Determine the current scanline from the (interpolated) scanline varying
    2. Determine the current bin from the fragment's x coordinate, and discard the fragment if the bin is empty
    3. For each shape in the bin potentially overlapping with the current fragment (check shape bounding box), compute the winding number and the minimum point-to-segment distance for the current fragment; if the current fragment is close enough to one of the shape's segments, blend the fragment color with the shape's stroke color; otherwise, if the current fragment is within the shape (odd winding number), blend the fragment color with the shape's fill color

The numbers of scanlines and of bins per scanline are chosen independently for each shape cloud, as they bound different costs of a fragment:

- **Scanline height** bounds the number of edges a fragment tests per shape, as edges are only assigned to scanlines. It is chosen so that a scanline holds a given number of edges (a render option) of a typical shape, plus the edges within the padding. The typical shape is determined by the median height per edge, weighted by shape area: fragments are distributed by area, so when zoomed in, a few large, detailed shapes among many small ones cover most of them.
- **Bin width** bounds the number of shapes a fragment considers. It is the median shape width times a bin width factor (a render option): bins about as wide as the shapes keep both the number of shapes per fragment and the number of bins each shape is listed in small.

The stroke width is a single uniform per shape cloud, shared by all of its shapes. Per-shape stroke widths are therefore not supported, as the bounding boxes and bins are computed without them. Scanline data is stored as `RGBA32UI` and the packed colors as `R32UI`, in textures of a fixed width. A shape cloud whose textures would need more lines than the GPU's maximum texture size is skipped with an error.

This approach has been chosen over a "standard approach" primarily to avoid CPU-side triangulation, reduce memory usage (no need to store triangles), enable thick outlines (strokes), allow for high-quality anti-aliasing, and for legacy (TissUUmaps 3) reasons.

|                                  | Standard approach                                                                                                                          | Compute shader approach                                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| **CPU load**                     | High (triangulation)                                                                                                                       | **Low** (edge table/bin construction)                                                                                |
| **Memory usage**                 | O(shapes \* triangles per shape \* 3)                                                                                                      | **O(shapes \* vertices per shape)**, but arbitrary limit wrt. how many vertices/shapes can overlap the same scanline |
| **Data transfer**                | Using element array buffers                                                                                                                | Using custom data textures                                                                                           |
| **GPU performance**              | Vertex shader transforms vertex coordinates in parallel; **cheap** fragment shader runs **for each fragment of each triangle** in parallel | Vertex shader does nothing; **VERY expensive** fragment shader runs **for each foreground pixel** in parallel        |
| **Outlines**                     | Difficult (separately triangulate outlines?)                                                                                               | **Easy** (thickness is efficiently defined within shader), but no arbitrary thickness                                |
| **Blending**                     | **Built-in alpha blending**, but requires explicit depth-sorting on the CPU                                                                | Explicitly computed, but still requires depth sorting on the CPU                                                     |
| **Anti-aliasing**                | Low quality (MSAA)                                                                                                                         | **High quality** (sub-pixel precision in distance calculation)                                                       |
| **Implementation / maintenance** | **Easy**                                                                                                                                   | Difficult                                                                                                            |

Standard approach (dismissed):

1. Triangulate shapes on the CPU, e.g. using [earcut](https://github.com/mapbox/earcut)
   - If necessary, only triangulate visible shapes --> quadtree?
   - If necessary, only re-triangulate when necessary (e.g. drawing)
2. Transfer triangle coordinates into a flat element array buffer on the GPU (cf. points rendering)
3. Call `gl.drawElements()` with `gl.TRIANGLES` and `count = # triangles`
   - The vertex shader will run once for every vertex and output shape-specific (flat) varyings such as `v_color`
   - The fragment shader will run once for every triangle fragment (i.e., overdraw will result in several runs for the same pixel)
