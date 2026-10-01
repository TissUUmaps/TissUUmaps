#version 300 es

#define MARKER_ATLAS_GRID_SIZE 4u
#define DISTANCE_MULTIPLIER 8.0
#define MAX_DISTANCE (127.5 / DISTANCE_MULTIPLIER) // in atlas texels
#define TEXTURE_LOD_BIAS -2.0
#define ALPHA_THRESHOLD 0.005

// https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices#implicit_defaults
precision highp float; // no default otherwise
precision highp int; // defaults to mediump otherwise
precision highp sampler2D; // defaults to lowp otherwise

// Signed distance fields (SDFs),
// pre-multiplied by DISTANCE_MULTIPLIER,
// and quantized into unsigned 8-bit integers
uniform sampler2D u_markerAtlas;

// Pick pass: every point is drawn in the color of its pick ID, see WebGLPicker
uniform bool u_pick;
uniform uint u_pickIdBase; // pick ID of the first point of this pass

uniform vec4 u_outlineRgba; // RGBA color to draw the marker in instead of its own, for the outline of the outlined point; alpha 0 for its own

flat in vec4 v_rgba;
flat in uvec3 v_marker;
flat in uint v_index;

out vec4 fragRgba;

// Packs a pick ID into the bytes of an RGBA8 target, 0xAABBGGRR (red in the lowest byte)
vec4 packPickId(uint id) {
    return vec4(float((id >> 0) & 0xFFu), float((id >> 8) & 0xFFu), float((id >> 16) & 0xFFu), float((id >> 24) & 0xFFu)) / 255.0;
}

void main() {
    vec2 uv = (vec2(v_marker.xy) + gl_PointCoord.xy) / float(MARKER_ATLAS_GRID_SIZE);
    float dist = (texture(u_markerAtlas, uv, TEXTURE_LOD_BIAS)[v_marker.z] - 0.5) * 255.0 / DISTANCE_MULTIPLIER;
    if(u_pick) {
        // the marker covers at least half of the fragment exactly where the
        // signed distance is non-negative, so picking needs no anti-aliasing
        if(dist < 0.0 || v_rgba.a < ALPHA_THRESHOLD) {
            discard;
        }
        fragRgba = packPickId(u_pickIdBase + v_index);
        return;
    }
    float pixelWidth = dFdx(uv.x) * float(textureSize(u_markerAtlas, 0).x); // atlas texels per device pixel
    // Beyond MAX_DISTANCE from the outline, the atlas stores the same distance for all texels.
    // Capping the anti-aliasing width keeps that background transparent at small point sizes.
    float aaWidth = min(pixelWidth, 2.0 * MAX_DISTANCE);
    float coverage = clamp(dist / aaWidth + 0.5, 0.0, 1.0);
    vec4 rgba = u_outlineRgba.a > 0.0 ? u_outlineRgba : v_rgba;
    rgba = vec4(rgba.rgb, 1.0) * (rgba.a * coverage); // premultiplied
    if(rgba.a < ALPHA_THRESHOLD) {
        discard;
    }
    fragRgba = rgba;
}
