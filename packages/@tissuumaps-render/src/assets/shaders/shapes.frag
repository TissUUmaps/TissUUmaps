#version 300 es

#define SCANLINE_DATA_TEXTURE_WIDTH 4096u  // see WebGLShapesRenderer
#define SHAPE_RGBAS_TEXTURE_WIDTH 4096u  // see WebGLShapesRenderer

// https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices#implicit_defaults
precision highp float; // no default otherwise
precision highp int; // defaults to mediump otherwise
precision highp sampler2D; // defaults to lowp otherwise
precision highp usampler2D; // no default otherwise

uniform uint u_numScanlines;
uniform uint u_numBins; // per scanline
uniform vec4 u_objectBounds; // (x, y, width, height), in data dimensions
uniform float u_opacityFactor; // layer and object opacity, 0 if the layer or object is invisible
uniform float u_halfStrokeWidth; // in data dimensions
uniform float u_devicePixelSize; // size of a device pixel, in data dimensions

/*
 * Scanline data (RGBA32UI texture)
 *
 * Memory layout: [bin table, shape refs, scanline 0, scanline 1, ..., scanline N]
 *   Bin table: [bin entry 0, bin entry 1, ...], two bin entries per texel, scanline-major
 *     Bin entry = (value offset of the bin's first shape ref, shape count M)
 *   Shape refs: [shape ref 0, shape ref 1, ...], four shape refs per texel
 *     Shape ref = texel offset of a shape of the bin's scanline
 *     (per bin: M consecutive shape refs, in the order the shapes are composited in)
 *   Scanline: [shape 1, shape 2, ...]
 *     Shape: [shape header, edge 1, edge 2, ..., edge L]
 *       Shape header = (shape index, edge count L, xmin, xmax)
 *       Edge = (x0, y0, x1, y1)
 *
 * Notes:
 * - Shape bounding boxes and edge vertices are in data dimensions
 * - Scanline data relates to the current shape cloud --> one draw call per data object
 * - Shape bounding boxes and the bin entries don't account for stroke widths, nor for the anti-aliased edges of fills
 *   and strokes, which reach half a pixel beyond them (this is why we cannot have shape-specific stroke widths, since
 *   strokes grow both inward and outward); shapes and edges are only padded on either side (see WebGLShapesRasterizer)
 * - Anti-aliasing is per shape, so edges shared by adjacent shapes are only partially covered by either,
 *   letting faint seams of the background show through between their fills (unless covered by opaque strokes)
 */
uniform usampler2D u_scanlineData;

// Shape fill RGBA colors (R32UI texture)
uniform usampler2D u_shapeFillRgbas;

// Shape stroke RGBA colors (R32UI texture)
uniform usampler2D u_shapeStrokeRgbas;

in vec2 v_pos; // in data dimensions
in float v_scanline; // in [0, u_numScanlines]

out vec4 fragRgba;

// fetches a texel from the given uint texture at a given offset
uvec4 utexel(usampler2D sampler, uint textureWidth, uint offset) {
    ivec2 p = ivec2(int(offset % textureWidth), int(offset / textureWidth));
    return texelFetch(sampler, p, 0);
}

// tests if a point p is left (>0), on (=0), or right (<0) of an infinite line through v0 and v1
// https://web.archive.org/web/20210506231426/http://geomalgorithms.com/a01-_area.html
float isPointLeftOfLine(vec2 p, vec2 v0, vec2 v1) {
    return (v1.x - v0.x) * (p.y - v0.y) - (p.x - v0.x) * (v1.y - v0.y);
}

// computes the minimum distance from point p to the line segment v0-v1
// https://web.archive.org/web/20210507021429/http://geomalgorithms.com/a02-_lines.html
float pointToSegmentDist(vec2 p, vec2 v0, vec2 v1) {
    vec2 v0ToP = p - v0;
    vec2 segment = v1 - v0;
    float segmentLength = length(segment); // strictly positive (zero-length edges are dropped in WebGLShapesRasterizer)
    vec2 unitSegment = segment / segmentLength;
    float pointOnSegment = dot(unitSegment, v0ToP);
    if(pointOnSegment <= 0.0) {
        return length(v0ToP);
    }
    if(pointOnSegment >= segmentLength) {
        return length(p - v1);
    }
    vec2 unitSegmentNormal = vec2(unitSegment.y, -unitSegment.x);
    float pointOnSegmentNormal = dot(unitSegmentNormal, v0ToP);
    return abs(pointOnSegmentNormal);
}

// computes the winding number for a given point p and n edges stored in data starting at offset
// https://web.archive.org/web/20210504233957/http://geomalgorithms.com/a03-_inclusion.html
int windingNumber(vec2 p, usampler2D sampler, uint textureWidth, uint offset, uint numEdges, out float minDist) {
    int wn = 0;
    minDist = 1e38;
    for(uint i = 0u; i < numEdges; ++i) {
        vec4 edge = uintBitsToFloat(utexel(sampler, textureWidth, offset + i));
        vec2 v0 = vec2(edge[0], edge[1]);
        vec2 v1 = vec2(edge[2], edge[3]);
        if(v0.y <= p.y) { // edge starts on/below point
            if(v1.y > p.y && isPointLeftOfLine(p, v0, v1) > 0.0) { // edge ends strictly above point, and point is strictly left of edge
                wn++;
            }
        } else { // edge starts strictly above point
            if(v1.y <= p.y && isPointLeftOfLine(p, v0, v1) < 0.0) { // edge ends on/below point, and point is strictly right of edge
                wn--;
            }
        }
        minDist = min(minDist, pointToSegmentDist(p, v0, v1));
    }
    return wn;
}

// Unpacks a uint-packed 8-bit RGBA color, 0xAABBGGRR (red in the lowest byte)
vec4 unpackRgba(uint rgba) {
    float r = float((rgba >> 0) & 0xFFu) / 255.0;
    float g = float((rgba >> 8) & 0xFFu) / 255.0;
    float b = float((rgba >> 16) & 0xFFu) / 255.0;
    float a = float((rgba >> 24) & 0xFFu) / 255.0;
    return vec4(r, g, b, a);
}

void main() {
    // anti-aliasing: strokes are at least one pixel wide, thinner strokes fade by coverage instead
    float hsw = max(u_halfStrokeWidth, 0.5 * u_devicePixelSize);
    float strokeFade = min(u_halfStrokeWidth / (0.5 * u_devicePixelSize), 1.0);
    float margin = hsw + 0.5 * u_devicePixelSize; // how far anti-aliased strokes and fills reach beyond the shapes
    if(v_pos.x < u_objectBounds[0] - margin || v_pos.x > u_objectBounds[0] + u_objectBounds[2] + margin || v_pos.y < u_objectBounds[1] - margin || v_pos.y > u_objectBounds[1] + u_objectBounds[3] + margin) {
        discard; // out of object bounds
    }
    // get the bin entry (clamp before converting: v_scanline and v_pos.x are out of range within the stroke margin)
    uint scanline = uint(clamp(v_scanline, 0.0, float(u_numScanlines - 1u)));
    uint bin = uint(clamp(float(u_numBins) * (v_pos.x - u_objectBounds[0]) / u_objectBounds[2], 0.0, float(u_numBins - 1u)));
    uint binIndex = scanline * u_numBins + bin;
    uvec4 binEntries2 = utexel(u_scanlineData, SCANLINE_DATA_TEXTURE_WIDTH, binIndex >> 1);
    uvec2 binEntry = (binIndex & 1u) == 0u ? binEntries2.xy : binEntries2.zw;
    uint shapeRefOffset = binEntry[0];
    uint numShapes = binEntry[1];
    if(numShapes == 0u) {
        discard; // no shapes on this scanline near the given x coordinate
    }
    // iterate over the bin's shapes
    fragRgba = vec4(0.0);
    uvec4 shapeRefs4;
    for(uint i = 0u; i < numShapes; ++i) {
        uint shapeRefIndex = shapeRefOffset + i;
        if(i == 0u || (shapeRefIndex & 3u) == 0u) {
            shapeRefs4 = utexel(u_scanlineData, SCANLINE_DATA_TEXTURE_WIDTH, shapeRefIndex >> 2);
        }
        uint shapeOffset = shapeRefs4[shapeRefIndex & 3u];
        uvec4 shapeInfo = utexel(u_scanlineData, SCANLINE_DATA_TEXTURE_WIDTH, shapeOffset);
        uint shapeIndex = shapeInfo[0];
        uint numEdges = shapeInfo[1];
        if(v_pos.x >= uintBitsToFloat(shapeInfo[2]) - margin && v_pos.x <= uintBitsToFloat(shapeInfo[3]) + margin) {
            float minDist;
            int wn = windingNumber(v_pos, u_scanlineData, SCANLINE_DATA_TEXTURE_WIDTH, shapeOffset + 1u, numEdges, minDist);
            // pixel coverage of the stroke and fill areas, approximated from the distance to the closest edge
            // (even-odd rule: holes are cut out whatever the orientation of their rings)
            float signedDist = (wn & 1) != 0 ? minDist : -minDist; // positive inside the fill area
            float fillCoverage = clamp(signedDist / u_devicePixelSize + 0.5, 0.0, 1.0);
            float strokeCoverage = strokeFade * clamp((hsw - minDist) / u_devicePixelSize + 0.5, 0.0, 1.0);
            // only fetch the colors of covered areas (most fragments are covered by the fill only)
            vec4 fillRgba = vec4(0.0);
            if(fillCoverage > 0.0) {
                uvec4 fillRgbaTexel = utexel(u_shapeFillRgbas, SHAPE_RGBAS_TEXTURE_WIDTH, shapeIndex);
                fillRgba = unpackRgba(fillRgbaTexel[0]);
                fillRgba.rgb = fillRgba.rgb * fillRgba.a; // premultiply
                fillRgba *= fillCoverage;
            }
            vec4 strokeRgba = vec4(0.0);
            if(strokeCoverage > 0.0) {
                uvec4 strokeRgbaTexel = utexel(u_shapeStrokeRgbas, SHAPE_RGBAS_TEXTURE_WIDTH, shapeIndex);
                strokeRgba = unpackRgba(strokeRgbaTexel[0]);
                strokeRgba.rgb = strokeRgba.rgb * strokeRgba.a; // premultiply
                strokeRgba *= strokeCoverage;
            }
            vec4 shapeRgba = strokeRgba + (1.0 - strokeRgba.a) * fillRgba;
            fragRgba = shapeRgba + (1.0 - shapeRgba.a) * fragRgba;
        }
    }
    // apply the layer and object opacity to the composited shapes (premultiplied)
    fragRgba *= u_opacityFactor;
}
