#version 300 es

// Marker atlas configuration
#define MARKER_ATLAS_GRID_SIZE 4u
#define N_MARKERS_PER_CHANNEL (MARKER_ATLAS_GRID_SIZE * MARKER_ATLAS_GRID_SIZE)

// Macro to discard the current vertex
#define DISCARD gl_PointSize = 0.0; gl_Position = vec4(2.0, 2.0, 0.0, 1.0); v_rgba = vec4(0.0); v_marker = uvec3(0); return;

// Global uniforms
uniform float u_globalPointSizeFactor;
uniform mat3x2 u_worldToViewportMatrix;
uniform vec2 u_viewportSize; // in world units
uniform vec2 u_canvasSize; // in device pixels

// Per-object uniforms
// An object is one point cloud on one layer, drawn in its own pass.
uniform mat3x2 u_dataToWorldMatrix;
uniform float u_pointSizeFactor; // converts a_size to world units
uniform float u_opacityFactor; // layer and object opacity, 0 if the layer or object is invisible

// Vertex attributes
layout(location = 0) in float a_x; // in data units
layout(location = 1) in float a_y; // in data units
layout(location = 2) in float a_size; // in the unit of the size configuration, see u_pointSizeFactor
layout(location = 3) in uint a_rgba; // packed 8-bit RGBA, 0xAABBGGRR (red in the lowest byte); alpha holds the point visibility and opacity
layout(location = 4) in uint a_marker; // marker index

// Outputs to fragment shader
flat out vec4 v_rgba; // with the opacity factor and coverage applied to alpha
flat out uvec3 v_marker; // (col, row, channel)

// Unpacks a uint-packed 8-bit RGBA color, 0xAABBGGRR (red in the lowest byte)
vec4 unpackRgba(uint rgba) {
    float r = float((rgba >> 0) & 0xFFu) / 255.0;
    float g = float((rgba >> 8) & 0xFFu) / 255.0;
    float b = float((rgba >> 16) & 0xFFu) / 255.0;
    float a = float((rgba >> 24) & 0xFFu) / 255.0;
    return vec4(r, g, b, a);
}

// Returns (col, row, channel) for a given marker index in the marker atlas
uvec3 markerAtlasCoords(uint marker) {
    uint col = (marker % N_MARKERS_PER_CHANNEL) % MARKER_ATLAS_GRID_SIZE;
    uint row = (marker % N_MARKERS_PER_CHANNEL) / MARKER_ATLAS_GRID_SIZE;
    uint channel = marker / N_MARKERS_PER_CHANNEL;
    return uvec3(col, row, channel);
}

// Main vertex shader function
void main() {
    // Compute point size in device pixels and discard points with non-positive size
    // The canvas is sized in device pixels already, so no device pixel ratio applies here.
    float canvasPixelRatio = dot(u_canvasSize / u_viewportSize, vec2(0.5)); // device pixels per world unit
    float worldPointSize = a_size * u_pointSizeFactor * u_globalPointSizeFactor;
    float devicePointSize = worldPointSize * canvasPixelRatio; // in device pixels
    if(devicePointSize <= 0.0) {
        DISCARD;
    }
    // Anti-aliasing: points are at least one device pixel wide, smaller points fade by (areal) coverage instead
    float coverage = min(devicePointSize, 1.0);
    gl_PointSize = max(devicePointSize, 1.0);

    // Compute point position in normalized device coordinates (NDCs)
    vec2 worldPosition = u_dataToWorldMatrix * vec3(a_x, a_y, 1.0);
    vec2 viewportPosition = u_worldToViewportMatrix * vec3(worldPosition, 1.0); // in [0, 1]
    vec2 ndcPosition = (2.0 * viewportPosition - 1.0) * vec2(1.0, -1.0); // in [-1, 1], y flipped
    gl_Position = vec4(ndcPosition, 0.0, 1.0);

    // Unpack color, apply the layer and object opacity and the coverage, and discard fully transparent points
    vec4 rgba = unpackRgba(a_rgba);
    rgba.a *= u_opacityFactor * coverage * coverage;
    if(rgba.a == 0.0) {
        DISCARD;
    }
    v_rgba = rgba;

    // Get marker atlas coordinates
    v_marker = markerAtlasCoords(a_marker);
}
