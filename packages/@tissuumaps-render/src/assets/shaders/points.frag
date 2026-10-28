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

flat in vec4 v_color;
flat in uvec3 v_marker;

out vec4 fragColor;

void main() {
    vec2 uv = (vec2(v_marker.xy) + gl_PointCoord.xy) / float(MARKER_ATLAS_GRID_SIZE);
    float dist = (texture(u_markerAtlas, uv, TEXTURE_LOD_BIAS)[v_marker.z] - 0.5) * 255.0 / DISTANCE_MULTIPLIER;
    float pixelWidth = dFdx(uv.x) * float(textureSize(u_markerAtlas, 0).x);
    // Beyond MAX_DISTANCE from the outline, the atlas stores the same distance for all texels.
    // Capping the anti-aliasing width keeps that background transparent at small point sizes.
    float aaWidth = min(pixelWidth, 2.0 * MAX_DISTANCE);
    float alpha = v_color.a * clamp(dist / aaWidth + 0.5, 0.0, 1.0);
    if(alpha < ALPHA_THRESHOLD) {
        discard;
    }
    fragColor = vec4(v_color.rgb * alpha, alpha);
}
