/**
* @file glUtils.js Utilities for WebGL-based marker drawing
* @author Fredrik Nysjo
* @see {@link glUtils}
*/

/**
 * @namespace glUtils
 * @property {Boolean} _initialized True when glUtils has been initialized
 */
glUtils = {
    _initialized: false,
    _options: {antialias: false, premultipliedAlpha: true, preserveDrawingBuffer: true},
    _markershapes: "misc/markershapes.png",

    // WebGL objects (both shared ones and ones that are per markerset UID)
    _programs: {},
    _buffers: {},
    _vaos: {},
    _textures: {},
    _query: null,
    _caps: {},

    // Marker rendering settings and info stored per UID (this could perhaps be
    // better handled by having an object per UID that stores all info and is
    // easy to delete when closing a marker tab...)
    _numPoints: {},              // {uid: numPoints, ...}
    _numEdges: {},               // {uid: numEdges, ...}
    _zOrder: {},                 // {uid: float, ...}
    _markerScalarRange: {},      // {uid: [minval, maxval], ...}
    _markerScalarPropertyName: {},  // {uid: string, ...}
    _markerScaleFactor: {},      // {uid: float, ...}
    _markerOpacity: {},          // {uid: alpha, ...}
    _markerBlendMode: {},        // {uid: string, ...}
    _markerStrokeWidth: {},      // {uid: float, ...}
    _markerFilled: {},           // {uid: boolean, ...}
    _markerOutline: {},          // {uid: boolean, ...}
    _useColorFromMarker: {},     // {uid: boolean, ...}
    _useColorFromColormap: {},   // {uid: boolean, ...}
    _useScaleFromMarker: {},     // {uid: boolean, ...}
    _useOpacityFromMarker: {},   // {uid: boolean, ...}
    _usePiechartFromMarker: {},  // {uid: boolean, ...}
    _useShapeFromMarker: {},     // {uid: boolean, ...}
    _useAbsoluteMarkerSize: {},  // {uid: boolean, ...}
    _piechartPalette: {},        // {uid: array or dict of colors, ...}
    _useSortByCol: {},           // {uid: boolean, ...}
    _colorscaleName: {},         // {uid: colorscaleName, ...}
    _colorscaleData: {},         // {uid: array of RGBA values, ...}
    _barcodeToLUTIndex: {},      // {uid: dict, ...}
    _barcodeToKey: {},           // {uid: dict, ...}
    _collectionItemIndex: {},    // {uid: number, ...}
    _markerInputsCached: {},     // {uid: dict, ...}
    _edgeInputsCached: {},       // {uid: dict, ...}

    // Global marker rendering settings and info
    _markerScale: 1.0,
    _useMarkerScaleFix: true,
    _globalMarkerScale: 1.0,
    _pickingEnabled: false,
    _pickingLocation: [0.0, 0.0],
    _pickedMarker: [-1, -1],
    _showColorbar: true,
    _showMarkerInfo: true,
    _resolutionScale: 1.0,        // If this is set to below 1.0, the WebGL output will be upscaled
    _resolutionScaleActual: 1.0,  // Automatic scaling factor computed from glUtils._resolutionScale
    _useInstancing: true,         // Use instancing and gl.TRIANGLE_STRIP to avoid size limit of gl.POINTS
    _showEdgesExperimental: true,
    _edgeThicknessRatio: 0.1,     // Ratio between edge thickness and marker size

    // Global region rendering settings and info
    _regionOpacity: 0.5,
    _regionStrokeWidth: 1.0,      // Base stroke width in pixels (larger values can give artifacts, so use with care!)
    _regionFillRule: "never",     // Possible values: "never" | "nonzero" | "oddeven"
    _regionShowOnTop: true,       // Draw regions on top of markers if this is true, otherwise under
    _regionUsePivotSplit: false,  // Use split edge lists for faster region rendering and less risk of overflow
    _regionUseColorByID: false,   // Map region object IDs to unique colors
    _regionDataSize: {},          // Size stored per region data texture and used for dynamic resizing
    _regionPicked: null,          // Key to regionUtils._regions dict, or null if no region is picked
    _regionMaxNumRegions: 524288, // Limit used for the LUT texture size (will be automatically increased if needed)

    // Other settings
    _logPerformance: false,       // Use GPU timer queries to log performance
    _piechartPaletteDefault: ["#fff100", "#ff8c00", "#e81123", "#ec008c", "#68217a", "#00188f", "#00bcf2", "#00b294", "#009e49", "#bad80a"]
}


glUtils._markersVS = `
    #define SHAPE_INDEX_CIRCLE 7.0
    #define SHAPE_INDEX_CIRCLE_NOSTROKE 16.0
    #define SHAPE_INDEX_GAUSSIAN 15.0
    #define SHAPE_GRID_SIZE 4.0
    #define MAX_NUM_IMAGES 192
    #define MAX_NUM_BARCODES 32768
    #define UV_SCALE 0.8
    #define SCALE_FIX (UV_SCALE / 0.7)  // For compatibility with old UV_SCALE
    #define DISCARD_VERTEX { gl_Position = vec4(2.0, 2.0, 2.0, 0.0); return; }

    uniform mat2 u_viewportTransform;
    uniform vec2 u_canvasSize;
    uniform int u_transformIndex;
    uniform float u_markerScale;
    uniform float u_globalMarkerScale;
    uniform vec2 u_markerScalarRange;
    uniform float u_markerOpacity;
    uniform float u_maxPointSize;
    uniform bool u_useColorFromMarker;
    uniform bool u_useColorFromColormap;
    uniform bool u_usePiechartFromMarker;
    uniform bool u_useShapeFromMarker;
    uniform bool u_useAbsoluteMarkerSize;
    uniform bool u_alphaPass;
    uniform int u_pickedMarker;
    uniform sampler2D u_colorLUT;
    uniform sampler2D u_colorscale;

    layout(std140) uniform TransformUniforms {
        mat2x4 imageToViewport[MAX_NUM_IMAGES];
    } u_transformUBO;

    layout(location = 0) in vec4 in_position;
    layout(location = 1) in int in_index;
    layout(location = 2) in float in_scale;
    layout(location = 3) in float in_shape;
    layout(location = 4) in float in_opacity;
    layout(location = 5) in float in_transform;

    flat out vec4 v_color;
    flat out vec2 v_shapeOrigin;
    flat out vec2 v_shapeSector;
    flat out float v_shapeIndex;
    flat out float v_shapeSize;
    #ifdef USE_INSTANCING
    out vec2 v_texCoord;
    #endif  // USE_INSTANCING

    vec3 hex_to_rgb(float v)
    {
        // Extract RGB color from 24-bit hex color stored in float
        v = clamp(v, 0.0, 16777215.0);
        return floor(mod((v + 0.49) / vec3(65536.0, 256.0, 1.0), 256.0)) / 255.0;
    }

    void main()
    {
        int transformIndex = u_transformIndex >= 0 ? u_transformIndex : int(in_transform);
        mat3x2 imageToViewport = mat3x2(transpose(u_transformUBO.imageToViewport[transformIndex]));
        vec2 viewportPos = imageToViewport * vec3(in_position.xy, 1.0);
        vec2 ndcPos = u_viewportTransform * ((viewportPos * 2.0 - 1.0) * vec2(1.0, -1.0));

        int lutIndex = int(mod(in_position.z, float(MAX_NUM_BARCODES)));
        v_color = texelFetch(u_colorLUT, ivec2(lutIndex & 4095, lutIndex >> 12), 0);

        if (u_useColorFromMarker || u_useColorFromColormap) {
            vec2 range = u_markerScalarRange;
            float normalized = (in_position.w - range[0]) / (range[1] - range[0]);
            v_color.rgb = texture(u_colorscale, vec2(normalized, 0.5)).rgb;
            if (u_useColorFromMarker) v_color.rgb = hex_to_rgb(in_position.w);
        }

        if (u_useShapeFromMarker && v_color.a > 0.0) {
            // Add one to marker index and normalize, to make things consistent
            // with how marker visibility and shape is stored in the LUT
            v_color.a = (floor(in_position.z / float(MAX_NUM_BARCODES)) + 1.0) / 255.0;
        }

        if (u_usePiechartFromMarker && v_color.a > 0.0) {
            v_shapeSector[0] = mod(in_shape, 4096.0) / 4095.0;
            v_shapeSector[1] = floor(in_shape / 4096.0) / 4095.0;
            v_color.rgb = hex_to_rgb(in_position.w);
            v_color.a = SHAPE_INDEX_CIRCLE_NOSTROKE / 255.0;
            if (u_pickedMarker == in_index) v_color.a = SHAPE_INDEX_CIRCLE / 255.0;

            // For the alpha pass, we only want to draw the marker once
            float sectorIndex = floor(in_position.z / float(MAX_NUM_BARCODES));
            if (u_alphaPass) v_color.a *= float(sectorIndex == 0.0);
        }

        gl_Position = vec4(ndcPos, 0.0, 1.0);

        if (u_useAbsoluteMarkerSize) {
            vec2 viewportPos2 = imageToViewport * vec3(in_position.xy + vec2(1.0, 0.0), 1.0);
            vec2 ndcPos2 = u_viewportTransform * ((viewportPos2 * 2.0 - 1.0) * vec2(1.0, -1.0));
            // When computing this scale factor, we want square markers with
            // unit size to match approx. one pixel in the image layer
            float imagePixelFactor = length((ndcPos2 - ndcPos) * u_canvasSize) * 0.68;
            gl_PointSize = (in_scale * u_markerScale * imagePixelFactor) * SCALE_FIX;
        } else {
            // Use default relative marker size
            gl_PointSize = (in_scale * u_markerScale * u_globalMarkerScale) * SCALE_FIX;
        }
        float alphaFactorSize = clamp(gl_PointSize, 0.2, 1.0); 
        gl_PointSize = clamp(gl_PointSize, 1.0, u_maxPointSize);

        v_shapeIndex = floor((v_color.a + 1e-5) * 255.0);
        v_shapeOrigin.x = mod(v_shapeIndex - 1.0, SHAPE_GRID_SIZE);
        v_shapeOrigin.y = floor((v_shapeIndex - 1.0) / SHAPE_GRID_SIZE);
        v_shapeSize = gl_PointSize;

    #ifdef USE_INSTANCING
        // Marker will be drawn as a triangle strip, so need to generate
        // texture coordinate and offset the output position depending on
        // which of the four corners we are processing
        v_texCoord = vec2(gl_VertexID & 1, (gl_VertexID >> 1) & 1);
        gl_Position.xy += (v_texCoord * 2.0 - 1.0) * (gl_PointSize / u_canvasSize);
        v_texCoord.y = 1.0 - v_texCoord.y;  // Flip Y-axis to match gl_PointCoord behaviour
    #endif  // USE_INSTANCING

        // Discard point here in vertex shader if marker is hidden
        v_color.a = v_color.a > 0.0 ? in_opacity * u_markerOpacity : 0.0;
        v_color.a *= alphaFactorSize * alphaFactorSize;
        v_color.a = clamp(v_color.a, 0.0, 1.0);
        if (v_color.a == 0.0) DISCARD_VERTEX;
    }
`;


glUtils._markersFS = `
    #define UV_SCALE 0.8
    #define SHAPE_INDEX_GAUSSIAN 15.0
    #define SHAPE_GRID_SIZE 4.0

    precision highp float;
    precision highp int;

    uniform float u_markerStrokeWidth;
    uniform bool u_markerFilled;
    uniform bool u_markerOutline;
    uniform bool u_usePiechartFromMarker;
    uniform bool u_alphaPass;
    uniform highp sampler2D u_shapeAtlas;

    flat in vec4 v_color;
    flat in vec2 v_shapeOrigin;
    flat in vec2 v_shapeSector;
    flat in float v_shapeIndex;
    flat in float v_shapeSize;
    #ifdef USE_INSTANCING
    in vec2 v_texCoord;
    #else
    #define v_texCoord gl_PointCoord
    #endif  // USE_INSTANCING

    layout(location = 0) out vec4 out_color;

    float sectorToAlpha(vec2 sector, vec2 uv)
    {
        vec2 dir = normalize(uv - 0.5);
        float theta = (atan(dir.x, dir.y) / 3.141592) * 0.5 + 0.5;
        return float(theta > sector[0] && theta < sector[1]);
    }

    float sectorToAlphaAA(vec2 sector, vec2 uv, float delta)
    {
        // This workaround avoids the problem with small pixel-wide
        // gaps that can appear between the first and last sector
        if (uv.y < 0.5 && abs(uv.x - 0.5) < delta) return 1.0;

        float accum = 0.0;
        accum += sectorToAlpha(sector, uv + vec2(-delta, -delta));
        accum += sectorToAlpha(sector, uv + vec2(delta, -delta));
        accum += sectorToAlpha(sector, uv + vec2(-delta, delta));
        accum += sectorToAlpha(sector, uv + vec2(delta, delta));
        return accum / 4.0;
    }

    void main()
    {
        vec2 uv = (v_texCoord - 0.5) * UV_SCALE + 0.5;
        uv = (uv + v_shapeOrigin) * (1.0 / SHAPE_GRID_SIZE);

        vec4 shapeColor = vec4(0.0);

        // Sample shape texture and reconstruct marker shape from signed
        // distance field (SDF) encoded in the red channel. Distance values
        // are assumed to be pre-multiplied by a scale factor 8.0 before
        // being quantized into 8-bit. Other channels in the texture are
        // currently ignored, but could be used for storing additional shapes
        // in the future!

        float pixelWidth = dFdx(uv.x) * float(textureSize(u_shapeAtlas, 0).x) * 8.0;
        float markerStrokeWidth = min(14.0, u_markerStrokeWidth) * 8.0;  // Keep within SDF range
        float distBias = u_markerFilled ? -pixelWidth * 0.25 : 0.0;  // Minification distance bias

        float distShape = (texture(u_shapeAtlas, uv, -2.0).r - 0.5) * 255.0;
        float distOutline = markerStrokeWidth - abs(distShape) + distBias;  // Add bias to fix darkening
        float alpha = clamp(distShape / pixelWidth + 0.5, 0.0, 1.0) * float(u_markerFilled);
        float alpha2 = clamp(distOutline / pixelWidth + 0.5, 0.0, 1.0) * float(u_markerOutline);
        if (distOutline < (markerStrokeWidth + 4.0) - 127.5) {
            alpha2 = 0.0;  // Fixes problem with alpha bleeding on minification
        }
        shapeColor = vec4(vec3(mix(1.0, 0.7, alpha2)), max(alpha, alpha2));
        if (!u_markerFilled && u_markerOutline) {
            shapeColor.rgb = vec3(1.0);  // Use brighter outline to show actual marker color 
        }

        // Handle special types of shapes (Gaussians and piecharts)

        if (v_shapeIndex == SHAPE_INDEX_GAUSSIAN) {
            shapeColor = vec4(vec3(1.0), smoothstep(0.5, 0.0, length(v_texCoord - 0.5)));
        }
        if (u_usePiechartFromMarker && !u_alphaPass) {
            float delta = 0.25 / v_shapeSize;
            shapeColor.a *= sectorToAlphaAA(v_shapeSector, v_texCoord, delta);
        }

        out_color = shapeColor * v_color;
        if (out_color.a < 0.004) discard;
    }
`;


glUtils._pickingVS = `
    #define UV_SCALE 0.8
    #define SCALE_FIX (UV_SCALE / 0.7)  // For compatibility with old UV_SCALE
    #define SHAPE_INDEX_CIRCLE_NOSTROKE 16.0
    #define SHAPE_GRID_SIZE 4.0
    #define MAX_NUM_IMAGES 192
    #define MAX_NUM_BARCODES 32768
    #define DISCARD_VERTEX { gl_Position = vec4(2.0, 2.0, 2.0, 0.0); return; }

    #define OP_CLEAR 0
    #define OP_WRITE_INDEX 1

    uniform mat2 u_viewportTransform;
    uniform vec2 u_canvasSize;
    uniform vec2 u_pickingLocation;
    uniform int u_transformIndex;
    uniform float u_markerScale;
    uniform float u_globalMarkerScale;
    uniform float u_markerOpacity;
    uniform float u_maxPointSize;
    uniform bool u_usePiechartFromMarker;
    uniform bool u_useShapeFromMarker;
    uniform bool u_useAbsoluteMarkerSize;
    uniform int u_op;
    uniform sampler2D u_colorLUT;
    uniform sampler2D u_shapeAtlas;

    layout(std140) uniform TransformUniforms {
        mat2x4 imageToViewport[MAX_NUM_IMAGES];
    } u_transformUBO;

    layout(location = 0) in vec4 in_position;
    layout(location = 1) in int in_index;
    layout(location = 2) in float in_scale;
    layout(location = 4) in float in_opacity;
    layout(location = 5) in float in_transform;

    flat out vec4 v_color;

    vec3 hex_to_rgb(float v)
    {
        // Extract RGB color from 24-bit hex color stored in float
        v = clamp(v, 0.0, 16777215.0);
        return floor(mod((v + 0.49) / vec3(65536.0, 256.0, 1.0), 256.0)) / 255.0;
    }

    void main()
    {
        int transformIndex = u_transformIndex >= 0 ? u_transformIndex : int(in_transform);
        mat3x2 imageToViewport = mat3x2(transpose(u_transformUBO.imageToViewport[transformIndex]));
        vec2 viewportPos = imageToViewport * vec3(in_position.xy, 1.0);
        vec2 ndcPos = u_viewportTransform * ((viewportPos * 2.0 - 1.0) * vec2(1.0, -1.0));

        v_color = vec4(0.0);
        if (u_op == OP_WRITE_INDEX) {
            int lutIndex = int(mod(in_position.z, float(MAX_NUM_BARCODES)));
            float shapeID = texelFetch(u_colorLUT, ivec2(lutIndex & 4095, lutIndex >> 12), 0).a;
            if (shapeID == 0.0) DISCARD_VERTEX;

            if (u_useShapeFromMarker) {
                // Add one to marker index and normalize, to make things consistent
                // with how marker visibility and shape is stored in the LUT
                shapeID = (floor(in_position.z / float(MAX_NUM_BARCODES)) + 1.0) / 255.0;
            }

            if (u_usePiechartFromMarker) {
                shapeID = SHAPE_INDEX_CIRCLE_NOSTROKE / 255.0;

                // For the picking pass, we only want to draw the marker once
                float sectorIndex = floor(in_position.z / float(MAX_NUM_BARCODES));
                if (sectorIndex > 0.0) DISCARD_VERTEX;
            }

            vec2 canvasPos = (ndcPos * 0.5 + 0.5) * u_canvasSize;
            canvasPos.y = (u_canvasSize.y - canvasPos.y);  // Y-axis is inverted

            float pointSize = 0.0;
            if (u_useAbsoluteMarkerSize) {
                vec2 viewportPos2 = imageToViewport * vec3(in_position.xy + vec2(1.0, 0.0), 1.0);
                vec2 ndcPos2 = u_viewportTransform * ((viewportPos2 * 2.0 - 1.0) * vec2(1.0, -1.0));
                // When computing this scale factor, we want square markers with
                // unit size to match approx. one pixel in the image layer
                float imagePixelFactor = length((ndcPos2 - ndcPos) * u_canvasSize) * 0.68;
                pointSize = (in_scale * u_markerScale * imagePixelFactor) * SCALE_FIX;
            } else {
                // Use default relative marker size
                pointSize = (in_scale * u_markerScale * u_globalMarkerScale) * SCALE_FIX;
            }
            pointSize = clamp(pointSize, 1.0, u_maxPointSize);

            // Do coarse inside/outside test against bounding box for marker
            vec2 uv = (canvasPos - u_pickingLocation) / pointSize + 0.5;
            uv.y = (1.0 - uv.y);  // Flip y-axis to match gl_PointCoord behaviour
            if (abs(uv.x - 0.5) > 0.5 || abs(uv.y - 0.5) > 0.5) DISCARD_VERTEX;

            // Do fine-grained inside/outside test by sampling the shape texture
            // with signed distance field (SDF) encoded in the red channel.
            // Currently, this does not take settings for fill and outline into
            // account, so all markers are assumed to be filled (TODO).
            vec2 shapeOrigin = vec2(0.0);
            shapeOrigin.x = mod((shapeID + 0.00001) * 255.0 - 1.0, SHAPE_GRID_SIZE);
            shapeOrigin.y = floor(((shapeID + 0.00001) * 255.0 - 1.0) / SHAPE_GRID_SIZE);
            uv = (uv - 0.5) * UV_SCALE + 0.5;
            uv = (uv + shapeOrigin) * (1.0 / SHAPE_GRID_SIZE);
            if (texture(u_shapeAtlas, uv).r < 0.5) DISCARD_VERTEX;

            // Also do a quick alpha-test to avoid picking non-visible markers
            if (in_opacity * u_markerOpacity <= 0.0) DISCARD_VERTEX

            // Output marker index encoded as hexadecimal color
            int encoded = in_index + 1;
            v_color.r = float((encoded >> 0) & 255) / 255.0;
            v_color.g = float((encoded >> 8) & 255) / 255.0;
            v_color.b = float((encoded >> 16) & 255) / 255.0;
            v_color.a = float((encoded >> 24) & 255) / 255.0;
        }

        gl_Position = vec4(-0.9999, -0.9999, 0.0, 1.0);
        gl_PointSize = 1.0;
    }
`;


glUtils._pickingFS = `
    precision highp float;
    precision highp int;

    flat in vec4 v_color;

    layout(location = 0) out vec4 out_color;

    void main()
    {
        out_color = v_color;
    }
`;


glUtils._edgesVS = `
    #define MAX_NUM_IMAGES 192

    uniform mat2 u_viewportTransform;
    uniform vec2 u_canvasSize;
    uniform int u_transformIndex;
    uniform float u_markerScale;
    uniform float u_globalMarkerScale;
    uniform float u_markerOpacity;
    uniform float u_maxPointSize;
    uniform float u_edgeThicknessRatio;
    uniform sampler2D u_colorLUT;

    layout(std140) uniform TransformUniforms {
        mat2x4 imageToViewport[MAX_NUM_IMAGES];
    } u_transformUBO;

    layout(location = 0) in vec4 in_position;
    layout(location = 1) in int in_index;
    layout(location = 4) in float in_opacity;
    layout(location = 5) in float in_transform;

    flat out vec4 v_color;
    out vec2 v_texCoord;

    void main()
    {
        int transformIndex0 = u_transformIndex >= 0 ? u_transformIndex : int(mod(in_transform, 256.0));
        int transformIndex1 = u_transformIndex >= 0 ? u_transformIndex : int(floor(in_transform / 256.0));
        mat3x2 imageToViewport0 = mat3x2(transpose(u_transformUBO.imageToViewport[transformIndex0]));
        mat3x2 imageToViewport1 = mat3x2(transpose(u_transformUBO.imageToViewport[transformIndex1]));

        vec2 localPos0 = in_position.xy;
        vec2 localPos1 = in_position.zw;

        // Transform 1st edge vertex
        vec2 viewportPos0 = imageToViewport0 * vec3(localPos0, 1.0);
        vec2 ndcPos0 = viewportPos0 * 2.0 - 1.0;
        ndcPos0.y = -ndcPos0.y;
        ndcPos0 = u_viewportTransform * ndcPos0;

        // Transform 2nd edge vertex
        vec2 viewportPos1 = imageToViewport1 * vec3(localPos1, 1.0);
        vec2 ndcPos1 = viewportPos1 * 2.0 - 1.0;
        ndcPos1.y = -ndcPos1.y;
        ndcPos1 = u_viewportTransform * ndcPos1;

        float pointSize = u_markerScale * u_globalMarkerScale;
        pointSize = clamp(pointSize, 0.05, u_maxPointSize);
        float lineThickness = max(0.5, u_edgeThicknessRatio * pointSize);
        float lineThicknessAdjusted = lineThickness + 0.25;  // Expanded thickness values,
        float lineThicknessAdjusted2 = lineThickness + 0.5;  // needed for anti-aliasing
        float lineThicknessOpacity = clamp(u_edgeThicknessRatio * pointSize, 0.005, 1.0);

        vec2 ndcMidpoint = (ndcPos1 + ndcPos0) * 0.5;
        vec2 ndcDeltaU = (ndcPos1 - ndcPos0) * 0.5;
        vec2 canvasDeltaU = ndcDeltaU * u_canvasSize;
        vec2 canvasDeltaV = vec2(-canvasDeltaU.y, canvasDeltaU.x);
        vec2 ndcDeltaV = lineThicknessAdjusted * normalize(canvasDeltaV) / u_canvasSize;

        gl_Position = vec4(ndcMidpoint, 0.0, 1.0);

        // Edge will be drawn as a triangle strip, so need to generate
        // texture coordinate and offset the output position depending on
        // which of the four corners we are processing
        v_texCoord = vec2(gl_VertexID & 1, (gl_VertexID >> 1) & 1);
        v_texCoord.y = ((v_texCoord.y - 0.5) * (lineThicknessAdjusted2 / lineThickness)) + 0.5;
        gl_Position.xy += (v_texCoord.x * 2.0 - 1.0) * ndcDeltaU;
        gl_Position.xy += (v_texCoord.y * 2.0 - 1.0) * ndcDeltaV;

        v_color.rgb = vec3(0.8);  // Use a fixed color (for now)
        v_color.a = in_opacity * u_markerOpacity * lineThicknessOpacity;
    }
`;


glUtils._edgesFS = `
    precision highp float;
    precision highp int;

    flat in vec4 v_color;
    in vec2 v_texCoord;

    layout(location = 0) out vec4 out_color;

    float subpixelCoverage(vec2 uv)
    {
        vec2 samples[4];  // Sample locations (from rotated grid)
        samples[0] = vec2(0.25, -0.75); samples[1] = vec2(0.75, 0.25);
        samples[2] = vec2(-0.25, 0.75); samples[3] = vec2(-0.75, -0.25);

        vec2 deltaX = dFdx(uv) * 0.5;
        vec2 deltaY = dFdy(uv) * 0.5;
        float accum = 0.0;
        for (int i = 0; i < 4; ++i) {
            // Check if sample is inside or outside the line for the edge
            vec2 uv_i = uv + samples[i].x * deltaX + samples[i].y * deltaY;
            bool inside = (uv_i.x > 0.0 && uv_i.x < 1.0 && uv_i.y > 0.0 && uv_i.y < 1.0);
            accum += float(inside);
        }
        return accum * (1.0 / 4.0);
    }

    void main()
    {
        out_color.rgb = v_color.rgb;
        out_color.a = v_color.a * subpixelCoverage(v_texCoord);
    }
`;


glUtils._regionsVS = `
    #define MAX_NUM_IMAGES 192

    uniform mat2 u_viewportTransform;
    uniform vec2 u_canvasSize;
    uniform int u_transformIndex;
    uniform vec4 u_imageBounds;
    uniform int u_numScanlines;

    layout(std140) uniform TransformUniforms {
        mat2x4 imageToViewport[MAX_NUM_IMAGES];
    } u_transformUBO;

    // Need to have attribute 0 enabled, otherwise some browsers (QtWebEngine)
    // will give performance warnings. It would otherwise have been simpler to
    // just compute the position/texcoord from gl_VertexID.
    layout(location = 0) in vec2 in_position;

    out vec2 v_texCoord;
    out vec2 v_localPos;
    out float v_scanline;
    flat out float v_pixelWidth;

    void main()
    {
        v_texCoord = in_position;
        v_scanline = v_texCoord.y * float(u_numScanlines);

        vec2 localPos;
        localPos.x = v_texCoord.x * u_imageBounds.z;
        localPos.y = v_texCoord.y * u_imageBounds.w;
        v_localPos = localPos;

        mat3x2 imageToViewport = mat3x2(transpose(u_transformUBO.imageToViewport[u_transformIndex]));
        vec2 viewportPos = imageToViewport * vec3(localPos, 1.0);
        vec2 ndcPos = viewportPos * 2.0 - 1.0;
        ndcPos.y = -ndcPos.y;
        ndcPos = u_viewportTransform * ndcPos;

        // Calculate pixel width in local coordinates. Need to do it here in the
        // vertex shader, because using pixel derivatives in the fragment shader
        // can cause broken stroke lines for large coordinates.
        vec2 ndcPos2 = ndcPos + 0.7 / u_canvasSize;
        mat2 viewportToLocal = inverse(u_viewportTransform * mat2(imageToViewport));
        v_pixelWidth = length(viewportToLocal * (ndcPos2 - ndcPos));

        gl_Position = vec4(ndcPos, 0.0, 1.0);
    }
`;


glUtils._regionsFS = `
    #define ALPHA 1.0
    #define STROKE_WIDTH 1.5
    #define STROKE_WIDTH_FILLED 1.0
    #define FILL_RULE_NEVER 0
    #define FILL_RULE_NONZERO 1
    #define FILL_RULE_ODDEVEN 2
    #define USE_OCCUPANCY_MASK 1
    #define SHOW_PIVOT_SPLIT_DEBUG 0
    #define SHOW_WORK_VISITED_BBOXES_DEBUG 0

    precision highp float;
    precision highp int;

    uniform int u_numScanlines;
    uniform float u_regionOpacity;
    uniform float u_regionStrokeWidth;
    uniform int u_regionFillRule;
    uniform int u_regionUsePivotSplit;
    uniform int u_regionUseColorByID;
    uniform highp sampler2D u_regionData;
    uniform highp sampler2D u_regionLUT;

    in vec2 v_texCoord;
    in vec2 v_localPos;
    in float v_scanline;
    flat in float v_pixelWidth;

    layout(location = 0) out vec4 out_color;

    float distPointToLine(vec2 p, vec2 v0, vec2 v1)
    {
        // Compute distance by first transforming the point p to the frame
        // defined by the line segment (v0, v1) and its normal vector. This
        // should be more robust and handle small distances to long line
        // segments better than just projecting the point onto the line.

        float a = length(v1 - v0);
        float b = length(p - v0);
        float c = length(p - v1);
        vec2 T = (v1 - v0) / (a + 1e-5);
        vec2 N = vec2(T.y, -T.x);
        vec2 t = mat2(T, N) * (p - v0);
        return (0.0 < t.x && t.x < a) ? abs(t.y) : min(b, c);
    }

    void main()
    {
        vec4 color = vec4(0.0);

        vec2 p = v_localPos;  // Current sample position
        int scanline = int(v_scanline);

        // float pixelWidth = length(dFdx(p.xy));  // Can cause precision problems!
        float pixelWidth = v_pixelWidth;  // Safer

        float strokeWidthPixels = u_regionStrokeWidth *
            (u_regionFillRule == FILL_RULE_NEVER ? STROKE_WIDTH : STROKE_WIDTH_FILLED);
        // For proper anti-aliasing, clamp stroke width to at least 1 pixel, and
        // make thinner strokes fade by coverage
        float strokeWidth = max(1.0, strokeWidthPixels) * pixelWidth;
        float strokeFade = min(1.0, strokeWidthPixels);

        float minEdgeDist = 1e7;  // Distance to closest edge

        vec4 scanlineInfo = texelFetch(u_regionData, ivec2(scanline, 0), 0);
        int offset = int(scanlineInfo.x) + 4096 * int(scanlineInfo.y);

    #if USE_OCCUPANCY_MASK
        // Do coarse empty space skipping first, by testing sample position against
        // occupancy bitmask stored in the first texel of the scanline.
        // Note: since the mask does not take the stroke width into account,
        // rendering thicker strokes with this enabled can result in artifacts.
        uvec4 maskData = uvec4(texelFetch(u_regionData, ivec2(offset & 4095, offset >> 12), 0));
        int bitIndex = int(v_texCoord.x * 63.9999);
        if ((maskData[bitIndex >> 4] & (1u << (bitIndex & 15))) == 0u) { discard; }
    #endif  // USE_OCCUPANCY_MASK

        float scanDir = 1.0;  // Can be positive or negative along the X-axis
        if (bool(u_regionUsePivotSplit)) {
            float pivot = texelFetch(u_regionData, ivec2((offset + 1) & 4095, (offset + 1) >> 12), 0).x;
            scanDir = p.x < pivot ? -1.0 : 1.0;
            if (p.x >= pivot) {
                scanlineInfo = texelFetch(u_regionData, ivec2(scanline + u_numScanlines, 0), 0);
                offset = int(scanlineInfo.x) + 4096 * int(scanlineInfo.y);
            }
        }

        offset += 2;  // Position pointer at first bounding box
        vec4 headerData = texelFetch(u_regionData, ivec2(offset & 4095, offset >> 12), 0);
        int objectID = int(headerData.z) - 1;
        int windingNumber = 0;
        int visitedBBoxes = 0;

        while (headerData.w != 0.0) {
            // Find next path with bounding box overlapping this sample position
            while (headerData.w != 0.0) {
                headerData = texelFetch(u_regionData, ivec2(offset & 4095, offset >> 12), 0);
                bool isPathBbox = headerData.z > 0.0;
                bool isClusterBbox = headerData.z == 0.0;
                visitedBBoxes += 1;

                if (headerData.x <= (p.x + strokeWidth) && (p.x - strokeWidth) <= headerData.y) {
                    if (isPathBbox) { break; }
                    if (isClusterBbox) { offset -= int(headerData.w); }
                }
                offset += int(headerData.w) + 1;
            }
            offset += 1;  // Position pointer at first edge element

            // Check if we are done for this object ID and need to update the color value
            if (objectID != int(headerData.z) - 1) {
                bool isInside = false;
                if (u_regionFillRule == FILL_RULE_NONZERO) { isInside = windingNumber != 0; }
                if (u_regionFillRule == FILL_RULE_ODDEVEN) { isInside = (windingNumber & 1) == 1; }

                if (isInside || minEdgeDist < strokeWidth) {
                    vec4 objectColor = texelFetch(u_regionLUT, ivec2(objectID & 4095, objectID >> 12), 0);
                    if (bool(u_regionUseColorByID)) {
                        // Map object ID to a unique color from low-discrepancy sequence
                        objectColor.rgb = fract(sqrt(vec3(2.0, 3.0, 5.0)) * float(objectID + 1));
                    }
                #if SHOW_PIVOT_SPLIT_DEBUG
                    if (bool(u_regionUsePivotSplit)) {
                        objectColor.rgb = scanDir > 0.0 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 1.0);
                    }
                #endif  // SHOW_PIVOT_SPLIT_DEBUG
                    float minEdgeDistSigned = isInside ? minEdgeDist : -minEdgeDist;
                    float strokeOpacity = smoothstep(strokeWidth, strokeWidth - pixelWidth, minEdgeDist) * strokeFade;
                    float fillOpacity = smoothstep(-pixelWidth, pixelWidth, minEdgeDistSigned) * u_regionOpacity;
                    objectColor.a *= clamp(strokeOpacity + fillOpacity, 0.0, 1.0);

                    color.a = objectColor.a + (1.0 - objectColor.a) * color.a;
                    color.rgb = mix(color.rgb, objectColor.rgb, objectColor.a);
                }

                windingNumber = 0;  // Reset intersection count
                minEdgeDist = 1e7;  // Reset distance to closest edge
                objectID = int(headerData.z) - 1;
            }

            // Do intersection tests with edge elements to update intersection count,
            // and also update the edge distance needed for outline rendering
            int count = int(headerData.w);
            for (int i = 0; i < count; ++i) {
                vec4 edgeData = texelFetch(u_regionData, ivec2((offset + i) & 4095, (offset + i) >> 12), 0);
                vec2 v0 = edgeData.xy;
                vec2 v1 = edgeData.zw;

                if (min(v0.y, v1.y) <= p.y && p.y < max(v0.y, v1.y)) {
                    float t = (p.y - v0.y) / (v1.y - v0.y + 1e-5);
                    float x = v0.x + (v1.x - v0.x) * t;
                    float weight = 0.0;
                    if (u_regionFillRule == FILL_RULE_NONZERO) { weight = sign(v1.y - v0.y); }
                    if (u_regionFillRule == FILL_RULE_ODDEVEN) { weight = 1.0; }
                    windingNumber += int(float((x - p.x) * scanDir > 0.0) * weight);
                }
                minEdgeDist = min(minEdgeDist, distPointToLine(p, v0, v1));
            }

            offset += count;
        }

        out_color = color;
        out_color.rgb /= max(1e-5, out_color.a);

    #if SHOW_WORK_VISITED_BBOXES_DEBUG
        {
            float t = clamp(float(visitedBBoxes) / 400.0, 0.0, 1.0) * 2.0;
            out_color.rgb = clamp(vec3(t - 1.0, 1.0 - abs(t - 1.0), 1.0 - t), 0.0, 1.0);
        }
    #endif  // SHOW_WORK_VISITED_BBOXES_DEBUG
    }
`;


glUtils._loadShaderProgram = function(gl, vertSource, fragSource, definitions="") {
    const vertShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertShader, "#version 300 es\n" + definitions + vertSource);
    gl.compileShader(vertShader);
    if (!gl.getShaderParameter(vertShader, gl.COMPILE_STATUS)) {
        console.log("Could not compile vertex shader: " + gl.getShaderInfoLog(vertShader));
    }

    const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragShader, "#version 300 es\n" + definitions + fragSource);
    gl.compileShader(fragShader);
    if (!gl.getShaderParameter(fragShader, gl.COMPILE_STATUS)) {
        console.log("Could not compile fragment shader: " + gl.getShaderInfoLog(fragShader));
    }

    const program = gl.createProgram();
    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);
    gl.deleteShader(vertShader);  // Flag shaders for automatic deletion after
    gl.deleteShader(fragShader);  // their program object is destroyed

    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.log("Unable to link shader program: " + gl.getProgramInfoLog(program));
        gl.deleteProgram(program);
        return null;
    }

    return program;
}


glUtils._createVertexBuffer = function(gl, numBytes) {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer); 
    gl.bufferData(gl.ARRAY_BUFFER, numBytes, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    return buffer;
}


glUtils._createIndexBuffer = function(gl, numBytes) {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer); 
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, numBytes, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    return buffer;
}


glUtils._createUniformBuffer = function(gl, numBytes=16384) {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, buffer); 
    gl.bufferData(gl.UNIFORM_BUFFER, numBytes, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);

    return buffer;
}


// Create a list of normalized sector angles in format (TODO)
glUtils._createPiechartAngles = function(sectors) {
    let angles = [], sum = 0.0;
    for (let i = 0; i < sectors.length; ++i) {
        sum += Number(sectors[i]);
    }
    for (let i = 0; i < sectors.length; ++i) {
        angles[i] = Number(sectors[i]) / sum;
    }
    for (let i = sectors.length - 2; i >= 0; --i) {
        angles[i] += angles[i + 1];
    }
    return angles;
}


/**
 * @summary Create WebGL resources and other objects for drawing marker dataset.
 * @param {String | Number} uid Identifier referencing the marker dataset in dataUtils.
 */
glUtils.loadMarkers = function(uid, forceUpdate) {
    if (!glUtils._initialized) return;
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl2", glUtils._options);

    let newInputs = {};  // Inputs that will require a vertex buffer update when changed

    // Get marker data and other info like image size
    const markerData = dataUtils.data[uid]["_processeddata"];
    const keyName = newInputs.keyName = dataUtils.data[uid]["_gb_col"];
    const xPosName = newInputs.xPosName = dataUtils.data[uid]["_X"];
    const yPosName = newInputs.yPosName = dataUtils.data[uid]["_Y"];
    let numPoints = markerData[xPosName].length;

    // If new marker data was loaded, we need to assign each barcode an index
    // that we can use with the LUT textures for color, visibility, etc.
    glUtils._updateBarcodeToLUTIndexDict(uid, markerData, keyName);
    const barcodeToLUTIndex = glUtils._barcodeToLUTIndex[uid];

    // Check how the user wants to draw the markers
    const colorPropertyName = newInputs.colorPropertyName = dataUtils.data[uid]["_cb_col"];
    const useColorFromMarker = newInputs.useColorFromMarker = (dataUtils.data[uid]["_cb_col"] != null && dataUtils.data[uid]["_cb_cmap"] == null);
    let hexColor = "#000000";

    const scalarPropertyName = newInputs.scalarPropertyName = dataUtils.data[uid]["_cb_col"];
    const colorscaleName = dataUtils.data[uid]["_cb_cmap"];
    const useColorFromColormap = newInputs.useColorFromColormap = dataUtils.data[uid]["_cb_cmap"] != null;
    let scalarRange = glUtils._markerScalarRange[uid];

    const scalePropertyName = newInputs.scalePropertyName = dataUtils.data[uid]["_scale_col"];
    const useScaleFromMarker = newInputs.useScaleFromMarker = dataUtils.data[uid]["_scale_col"] != null;
    const markerScaleFactor = dataUtils.data[uid]["_scale_factor"];
    const markerCoordFactor = newInputs.markerCoordFactor = dataUtils.data[uid]["_coord_factor"];
    const useAbsoluteMarkerSize = glUtils._useAbsoluteMarkerSize[uid] != undefined ? glUtils._useAbsoluteMarkerSize[uid] : false;  // TODO
    
    const sectorsPropertyName = newInputs.sectorsPropertyName = dataUtils.data[uid]["_pie_col"];
    const usePiechartFromMarker = dataUtils.data[uid]["_pie_col"] != null;
    let piechartPalette = [...glUtils._piechartPaletteDefault];
    if (dataUtils.data[uid]["_pie_dict"] && sectorsPropertyName) {
        const sectorNames = sectorsPropertyName.split(";");
        for (let i = 0; i < sectorNames.length; ++i) {
            const key = (Array.isArray(dataUtils.data[uid]["_pie_dict"]))?i:sectorNames[i];
            console.log(key, i, dataUtils.data[uid]["_pie_dict"])
            if (dataUtils.data[uid]["_pie_dict"].hasOwnProperty(key)) {
                piechartPalette[i] = dataUtils.data[uid]["_pie_dict"][key];
            }
        }
    }
    newInputs.piechartPalette = piechartPalette;
    const numSectors = usePiechartFromMarker ? markerData[sectorsPropertyName][0].toString().split(";").length : 1;

    const shapePropertyName = newInputs.shapePropertyName = dataUtils.data[uid]["_shape_col"];
    const useShapeFromMarker = newInputs.useShapeFromMarker = dataUtils.data[uid]["_shape_col"] != null;
    const numShapes = Object.keys(markerUtils._symbolStrings).length;
    let shapeIndex = 0;

    const opacityPropertyName = newInputs.opacityPropertyName = dataUtils.data[uid]["_opacity_col"];
    const useOpacityFromMarker = newInputs.useOpacityFromMarker = dataUtils.data[uid]["_opacity_col"] != null;
    const markerOpacityFactor = dataUtils.data[uid]["_opacity"];
    const markerBlendMode = glUtils._markerBlendMode[uid] != undefined ? glUtils._markerBlendMode[uid] : "over";  // TODO

    const markerStrokeWidth = dataUtils.data[uid]["_stroke_width"];
    const markerFilled = !dataUtils.data[uid]["_no_fill"];
    const markerOutline = !dataUtils.data[uid]["_no_outline"];

    const collectionItemPropertyName = newInputs.collectionItemPropertyName = dataUtils.data[uid]["_collectionItem_col"];
    const useCollectionItemFromMarker = newInputs.useCollectionItemFromMarker = dataUtils.data[uid]["_collectionItem_col"] != null;
    const collectionItemFixed = newInputs.collectionItemFixed = dataUtils.data[uid]["_collectionItem_fixed"];
    let collectionItemIndex = collectionItemFixed;

    const sortByCol = newInputs.sortByCol = dataUtils.data[uid]["_sortby_col"];
    const useSortByCol = newInputs.useSortByCol = dataUtils.data[uid]["_sortby_col"] != null;
    const sortByDesc = newInputs.sortByDesc = dataUtils.data[uid]["_sortby_desc"];

    const zOrder = dataUtils.data[uid]["_z_order"];

    // Additional info about the vertex format. Make sure to update also
    // NUM_BYTES_PER_MARKER and NUM_BYTES_PER_MARKER_SECONDARY when making
    // changes to the format! Two buffers will be used for the vertex data to
    // avoid the 1GB max size limit imposed by QtWebEngine/Chromium. Also
    // remember to update glUtils._updateBindingOffsetsForVAO if changing the
    // format.
    const NUM_BYTES_PER_MARKER = 16;
    const NUM_BYTES_PER_MARKER_SECONDARY = 16;
    const POINT_OFFSET = numPoints * 0,
          INDEX_OFFSET = numPoints * 0,
          SCALE_OFFSET = numPoints * 4,
          SHAPE_OFFSET = numPoints * 8,
          OPACITY_OFFSET = numPoints * 12,
          TRANSFORM_OFFSET = numPoints * 14;
    const POINT_LOCATION = 0,
          INDEX_LOCATION = 1,
          SCALE_LOCATION = 2,
          SHAPE_LOCATION = 3,
          OPACITY_LOCATION = 4,
          TRANSFORM_LOCATION = 5;

    const lastInputs = glUtils._markerInputsCached[uid];
    if (forceUpdate || (lastInputs != JSON.stringify(newInputs)) || dataUtils.data[uid]["modified"]) {
        dataUtils.data[uid]["modified"] = false;
        forceUpdate = true;
        scalarRange = [1e9, -1e9];  // This range will be computed from the data

        console.time("Generate index data");
        const numIndices = numPoints * numSectors;
        let indicesSorted = new Uint32Array(numIndices);
        {
            for (let index = 0; index < numIndices; ++index) {
                indicesSorted[index] = index;
            }
            if (useSortByCol) {
                const colData = markerData[sortByCol];
                if (sortByDesc) {
                    // Sort in descending order
                    indicesSorted.sort((i, j) => Number(colData[Math.floor(j / numSectors)]) -
                        Number(colData[Math.floor(i / numSectors)]));
                } else {  // Sort in ascending order
                    indicesSorted.sort((i, j) => Number(colData[Math.floor(i / numSectors)]) -
                        Number(colData[Math.floor(j / numSectors)]));
                }
            }
        }
        console.timeEnd("Generate index data");

        // Extract and upload vertex data for markers. For datasets with tens of of
        // millions of points, the vertex data can be quite large, so we upload the
        // data in chunks to the GPU buffer to avoid having to allocate a large
        // temporary buffer in system memory.
        console.time("Generate vertex data");
        let chunkSize = 100000;
        for (let offset = 0; offset < numPoints; offset += chunkSize) {
            // Allocate space for vertex data that will be uploaded to vertex buffer
            if (offset + chunkSize >= numPoints) chunkSize = numPoints - offset;
            // console.log(offset, chunkSize, numPoints);
            let bytedata_point = new Float32Array(chunkSize * 4);
            let bytedata_index = new Int32Array(chunkSize * 1);
            let bytedata_scale = new Float32Array(chunkSize * 1);
            let bytedata_shape = new Float32Array(chunkSize * 1);
            let bytedata_opacity = new Uint16Array(chunkSize * 1);
            let bytedata_transform = new Uint16Array(chunkSize * 1);

            if (usePiechartFromMarker) {
                // For piecharts, we need to create one marker per piechart sector,
                // so also have to allocate additional space for the vertex data
                bytedata_point = new Float32Array(chunkSize * numSectors * 4);
                bytedata_index = new Int32Array(chunkSize * numSectors * 1);
                bytedata_scale = new Float32Array(chunkSize * numSectors * 1);
                bytedata_shape = new Float32Array(chunkSize * numSectors * 1);
                bytedata_opacity = new Uint16Array(chunkSize * numSectors * 1);
                bytedata_transform = new Uint16Array(chunkSize * numSectors * 1);

                for (let i = 0; i < chunkSize; ++i) {
                    const markerIndex = indicesSorted[i + offset];
                    const sectors = markerData[sectorsPropertyName][markerIndex].toString().split(";");
                    const piechartAngles = glUtils._createPiechartAngles(sectors);
                    const lutIndex = (keyName != null) ? barcodeToLUTIndex[markerData[keyName][markerIndex]] : 0;
                    const opacity = useOpacityFromMarker ? markerData[opacityPropertyName][markerIndex] : 1.0;
                    if (useCollectionItemFromMarker) collectionItemIndex = markerData[collectionItemPropertyName][markerIndex];

                    for (let j = 0; j < numSectors; ++j) {
                        const k = (i * numSectors + j);
                        const sectorIndex = j;
                        hexColor = piechartPalette[j % piechartPalette.length];
                        
                        bytedata_point[4 * k + 0] = markerData[xPosName][markerIndex] * markerCoordFactor;
                        bytedata_point[4 * k + 1] = markerData[yPosName][markerIndex] * markerCoordFactor;
                        bytedata_point[4 * k + 2] = lutIndex + sectorIndex * 32768.0;
                        bytedata_point[4 * k + 3] = Number("0x" + hexColor.substring(1,7));
                        bytedata_index[k] = markerIndex;  // Store index needed for picking
                        bytedata_scale[k] = useScaleFromMarker ? markerData[scalePropertyName][markerIndex] : 1.0;
                        bytedata_shape[k] =
                            Math.floor((j < numSectors - 1 ? piechartAngles[j + 1] : 0.0) * 4095.0) +
                            Math.floor(piechartAngles[j] * 4095.0) * 4096.0;
                        bytedata_opacity[k] = Math.floor(Math.max(0.0, Math.min(1.0, opacity)) * 65535.0);
                        bytedata_transform[k] = collectionItemIndex;
                    }
                }
            } else {
                for (let i = 0; i < chunkSize; ++i) {
                    const markerIndex = indicesSorted[i + offset];
                    const lutIndex = (keyName != null) ? barcodeToLUTIndex[markerData[keyName][markerIndex]] : 0;
                    const opacity = useOpacityFromMarker ? markerData[opacityPropertyName][markerIndex] : 1.0;
                    if (useCollectionItemFromMarker) collectionItemIndex = markerData[collectionItemPropertyName][markerIndex];

                    if (useColorFromMarker) hexColor = markerData[colorPropertyName][markerIndex];
                    if (useColorFromColormap) {
                        scalarValue = markerData[scalarPropertyName][markerIndex];
                        // Update scalar range that will be used for normalizing the values
                        scalarRange[0] = Math.min(scalarRange[0], isNaN(scalarValue) ? Infinity : scalarValue);
                        scalarRange[1] = Math.max(scalarRange[1], isNaN(scalarValue) ? -Infinity : scalarValue);
                    }
                    if (useShapeFromMarker) {
                        shapeIndex = markerData[shapePropertyName][markerIndex];
                        // Check if shapeIndex is a symbol names that needs to be converted to an index
                        if (isNaN(shapeIndex)) shapeIndex = markerUtils._symbolStrings.indexOf(shapeIndex);
                        shapeIndex = Math.max(0.0, Math.floor(Number(shapeIndex))) % numShapes;
                    }

                    bytedata_point[4 * i + 0] = markerData[xPosName][markerIndex] * markerCoordFactor;
                    bytedata_point[4 * i + 1] = markerData[yPosName][markerIndex] * markerCoordFactor;
                    bytedata_point[4 * i + 2] = lutIndex + Number(shapeIndex) * 32768.0;
                    bytedata_point[4 * i + 3] = useColorFromColormap ? Number(scalarValue)
                                                                     : Number("0x" + hexColor.substring(1,7));
                    bytedata_index[i] = markerIndex;  // Store index needed for picking
                    bytedata_scale[i] = useScaleFromMarker ? markerData[scalePropertyName][markerIndex] : 1.0;
                    bytedata_opacity[i] = Math.floor(Math.max(0.0, Math.min(1.0, opacity)) * 65535.0);
                    bytedata_transform[i] = collectionItemIndex;
                }
            }

            if (!(uid + "_markers" in glUtils._buffers)) {
                document.getElementById(uid + "_menu-UI").addEventListener("input", glUtils.updateColorLUTTextures);
                document.getElementById(uid + "_menu-UI").addEventListener("input", glUtils.draw);
            }

            // Create WebGL objects (if this has not already been done)
            if (!(uid + "_markers" in glUtils._buffers))
                glUtils._buffers[uid + "_markers"] = glUtils._createVertexBuffer(
                    gl, numPoints * numSectors * NUM_BYTES_PER_MARKER);
            if (!(uid + "_markers_secondary" in glUtils._buffers))
                glUtils._buffers[uid + "_markers_secondary"] = glUtils._createVertexBuffer(
                    gl, numPoints * numSectors * NUM_BYTES_PER_MARKER_SECONDARY);
            if (!(uid + "_markers" in glUtils._vaos))
                glUtils._vaos[uid + "_markers"] = gl.createVertexArray();
            if (!(uid + "_markers_instanced" in glUtils._vaos))
                glUtils._vaos[uid + "_markers_instanced"] = gl.createVertexArray();
            if (!(uid + "_colorLUT" in glUtils._textures))
                glUtils._textures[uid + "_colorLUT"] = glUtils._createColorLUTTexture(gl);
            if (!(uid + "_colorscale" in glUtils._textures))
                glUtils._textures[uid + "_colorscale"] = glUtils._createColorScaleTexture(gl);

            // Upload chunks of vertex data to buffer
            if (offset == 0) {
                // If the number of sectors used is changed, we have to reallocate the buffers
                {
                    gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_markers"]);
                    const newBufferSize = numPoints * numSectors * NUM_BYTES_PER_MARKER;
                    const oldBufferSize = gl.getBufferParameter(gl.ARRAY_BUFFER, gl.BUFFER_SIZE);
                    if (newBufferSize != oldBufferSize)
                        gl.bufferData(gl.ARRAY_BUFFER, newBufferSize, gl.STATIC_DRAW);
                    gl.bindBuffer(gl.ARRAY_BUFFER, null);
                }
                {
                    gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_markers_secondary"]);
                    const newBufferSize = numPoints * numSectors * NUM_BYTES_PER_MARKER_SECONDARY;
                    const oldBufferSize = gl.getBufferParameter(gl.ARRAY_BUFFER, gl.BUFFER_SIZE);
                    if (newBufferSize != oldBufferSize)
                        gl.bufferData(gl.ARRAY_BUFFER, newBufferSize, gl.STATIC_DRAW);
                    gl.bindBuffer(gl.ARRAY_BUFFER, null);
                }
            }
            gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_markers"]);
            gl.bufferSubData(gl.ARRAY_BUFFER, (POINT_OFFSET + offset * 16) * numSectors, bytedata_point);
            gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_markers_secondary"]);
            gl.bufferSubData(gl.ARRAY_BUFFER, (INDEX_OFFSET + offset * 4) * numSectors, bytedata_index);
            gl.bufferSubData(gl.ARRAY_BUFFER, (SCALE_OFFSET + offset * 4) * numSectors, bytedata_scale);
            gl.bufferSubData(gl.ARRAY_BUFFER, (SHAPE_OFFSET + offset * 4) * numSectors, bytedata_shape);
            gl.bufferSubData(gl.ARRAY_BUFFER, (OPACITY_OFFSET + offset * 2) * numSectors, bytedata_opacity);
            gl.bufferSubData(gl.ARRAY_BUFFER, (TRANSFORM_OFFSET + offset * 2) * numSectors, bytedata_transform);
            gl.bindBuffer(gl.ARRAY_BUFFER, null);
        }
        console.timeEnd("Generate vertex data");

        // Set up VAO with vertex format for drawing
        gl.bindVertexArray(glUtils._vaos[uid + "_markers"]);
        gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_markers"]);
        gl.enableVertexAttribArray(POINT_LOCATION);
        gl.vertexAttribPointer(POINT_LOCATION, 4, gl.FLOAT, false, 0, POINT_OFFSET * numSectors);
        gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_markers_secondary"]);
        gl.enableVertexAttribArray(INDEX_LOCATION);
        gl.vertexAttribIPointer(INDEX_LOCATION, 1, gl.INT, 0, INDEX_OFFSET * numSectors);
        gl.enableVertexAttribArray(SCALE_LOCATION);
        gl.vertexAttribPointer(SCALE_LOCATION, 1, gl.FLOAT, false, 0, SCALE_OFFSET * numSectors);
        gl.enableVertexAttribArray(SHAPE_LOCATION);
        gl.vertexAttribPointer(SHAPE_LOCATION, 1, gl.FLOAT, false, 0, SHAPE_OFFSET * numSectors);
        gl.enableVertexAttribArray(OPACITY_LOCATION);
        gl.vertexAttribPointer(OPACITY_LOCATION, 1, gl.UNSIGNED_SHORT, true, 0, OPACITY_OFFSET * numSectors);
        gl.enableVertexAttribArray(TRANSFORM_LOCATION);
        gl.vertexAttribPointer(TRANSFORM_LOCATION, 1, gl.UNSIGNED_SHORT, false, 0, TRANSFORM_OFFSET * numSectors);
        gl.bindVertexArray(null);

        // Set up 2nd VAO (for experimental instanced drawing)
        gl.bindVertexArray(glUtils._vaos[uid + "_markers_instanced"]);
        gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_markers"]);
        gl.enableVertexAttribArray(POINT_LOCATION);
        gl.vertexAttribPointer(POINT_LOCATION, 4, gl.FLOAT, false, 0, POINT_OFFSET * numSectors);
        gl.vertexAttribDivisor(POINT_LOCATION, 1);
        gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_markers_secondary"]);
        gl.enableVertexAttribArray(INDEX_LOCATION);
        gl.vertexAttribIPointer(INDEX_LOCATION, 1, gl.INT, 0, INDEX_OFFSET * numSectors);
        gl.vertexAttribDivisor(INDEX_LOCATION, 1);
        gl.enableVertexAttribArray(SCALE_LOCATION);
        gl.vertexAttribPointer(SCALE_LOCATION, 1, gl.FLOAT, false, 0, SCALE_OFFSET * numSectors);
        gl.vertexAttribDivisor(SCALE_LOCATION, 1);
        gl.enableVertexAttribArray(SHAPE_LOCATION);
        gl.vertexAttribPointer(SHAPE_LOCATION, 1, gl.FLOAT, false, 0, SHAPE_OFFSET * numSectors);
        gl.vertexAttribDivisor(SHAPE_LOCATION, 1);
        gl.enableVertexAttribArray(OPACITY_LOCATION);
        gl.vertexAttribPointer(OPACITY_LOCATION, 1, gl.UNSIGNED_SHORT, true, 0, OPACITY_OFFSET * numSectors);
        gl.vertexAttribDivisor(OPACITY_LOCATION, 1);
        gl.enableVertexAttribArray(TRANSFORM_LOCATION);
        gl.vertexAttribPointer(TRANSFORM_LOCATION, 1, gl.UNSIGNED_SHORT, false, 0, TRANSFORM_OFFSET * numSectors);
        gl.vertexAttribDivisor(TRANSFORM_LOCATION, 1);
        gl.bindVertexArray(null);
    }
    glUtils._markerInputsCached[uid] = JSON.stringify(newInputs);

    // Generate separate WebGL resources for drawing graph edges (if markerset
    // contains spatial connectivity data and the user wants to display it)
    const numEdges = glUtils._loadEdges(uid, forceUpdate);

    // Update marker info and LUT + colormap textures
    glUtils._numPoints[uid] = numPoints * numSectors;
    glUtils._numEdges[uid] = numEdges;
    glUtils._zOrder[uid] = zOrder;
    glUtils._markerScalarRange[uid] = scalarRange;
    glUtils._markerScalarPropertyName[uid] = scalarPropertyName;
    glUtils._markerScaleFactor[uid] = markerScaleFactor;
    glUtils._markerOpacity[uid] = markerOpacityFactor;
    glUtils._markerBlendMode[uid] = markerBlendMode;
    glUtils._markerStrokeWidth[uid] = markerStrokeWidth;
    glUtils._markerFilled[uid] = markerFilled;
    glUtils._markerOutline[uid] = markerOutline;
    glUtils._useColorFromMarker[uid] = useColorFromMarker;
    glUtils._useColorFromColormap[uid] = useColorFromColormap;
    glUtils._useScaleFromMarker[uid] = useScaleFromMarker;
    glUtils._useOpacityFromMarker[uid] = useOpacityFromMarker;
    glUtils._usePiechartFromMarker[uid] = usePiechartFromMarker;
    glUtils._useShapeFromMarker[uid] = useShapeFromMarker;
    glUtils._useAbsoluteMarkerSize[uid] = useAbsoluteMarkerSize;
    glUtils._piechartPalette[uid] = piechartPalette;
    glUtils._useSortByCol[uid] = useSortByCol;
    glUtils._colorscaleName[uid] = colorscaleName;
    glUtils._collectionItemIndex[uid] = collectionItemFixed;
    if (useColorFromColormap) {
        glUtils._updateColorScaleTexture(gl, uid, glUtils._textures[uid + "_colorscale"]);
    }
    glUtils._updateColorbarCanvas();
    glUtils._updateColorLUTTexture(gl, uid, glUtils._textures[uid + "_colorLUT"]);
    markerUtils.updatePiechartLegend();
}


/**
 * @summary Delete WebGL resources and other objects created for drawing marker dataset.
 * @param {String | Number} uid Identifier referencing the marker dataset in dataUtils.
 */
glUtils.deleteMarkers = function(uid) {
    if (!glUtils._initialized) return;
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl2", glUtils._options);

    if (!(uid in glUtils._numPoints)) return;  // Assume markers are already deleted

    // Delete marker settings and info for UID
    delete glUtils._numPoints[uid];
    delete glUtils._numEdges[uid];
    delete glUtils._zOrder[uid];
    delete glUtils._markerScaleFactor[uid];
    delete glUtils._markerScalarRange[uid];
    delete glUtils._markerScalarPropertyName[uid];
    delete glUtils._markerOpacity[uid];
    delete glUtils._markerBlendMode[uid];
    delete glUtils._markerStrokeWidth[uid];
    delete glUtils._markerFilled[uid];
    delete glUtils._markerOutline[uid];
    delete glUtils._useColorFromMarker[uid];
    delete glUtils._useColorFromColormap[uid];
    delete glUtils._useScaleFromMarker[uid];
    delete glUtils._useOpacityFromMarker[uid];
    delete glUtils._usePiechartFromMarker[uid];
    delete glUtils._useShapeFromMarker[uid];
    delete glUtils._useAbsoluteMarkerSize[uid];
    delete glUtils._piechartPalette[uid];
    delete glUtils._useSortByCol[uid];
    delete glUtils._colorscaleName[uid];
    delete glUtils._colorscaleData[uid];
    delete glUtils._barcodeToLUTIndex[uid];
    delete glUtils._barcodeToKey[uid];
    delete glUtils._collectionItemIndex[uid];
    delete glUtils._markerInputsCached[uid];
    delete glUtils._edgeInputsCached[uid];

    // Clean up WebGL resources
    gl.deleteBuffer(glUtils._buffers[uid + "_markers"]);
    gl.deleteBuffer(glUtils._buffers[uid + "_markers_secondary"]);
    gl.deleteVertexArray(glUtils._vaos[uid + "_markers"]);
    gl.deleteVertexArray(glUtils._vaos[uid + "_markers_instanced"]);
    gl.deleteBuffer(glUtils._buffers[uid + "_edges"]);
    gl.deleteVertexArray(glUtils._vaos[uid + "_edges"]);
    gl.deleteTexture(glUtils._textures[uid + "_colorLUT"]);
    gl.deleteTexture(glUtils._textures[uid + "_colorscale"]);
    delete glUtils._buffers[uid + "_markers"];
    delete glUtils._buffers[uid + "_markers_secondary"];
    delete glUtils._vaos[uid + "_markers"];
    delete glUtils._vaos[uid + "_markers_instanced"];
    delete glUtils._buffers[uid + "_edges"];
    delete glUtils._vaos[uid + "_edges"];
    delete glUtils._textures[uid + "_colorLUT"];
    delete glUtils._textures[uid + "_colorscale"];
    // Make sure colorbar is also deleted from the 2D canvas
    glUtils._updateColorbarCanvas();

    // Make sure piechart legend is deleted if it was used for this UID
    markerUtils.updatePiechartLegend();
}


glUtils._updateBindingOffsetsForCurrentMarkerVAO = function(gl, uid, offset, numPoints) {
    // This function is used for updating the offsets of vertex bindings for
    // instanced drawing. This is only necessary because drawArrayInstanced and
    // drawElementsInstanced ignores any offset/first parameters we provide
    // for instanced arrays (arrays with non-zero vertex attrib divisor).

    // Additional info about the vertex format. Make sure to update these values
    // if/when also changing the vertex format in glUtils.loadMarkers!
    const POINT_OFFSET = numPoints * 0,
          INDEX_OFFSET = numPoints * 0,
          SCALE_OFFSET = numPoints * 4,
          SHAPE_OFFSET = numPoints * 8,
          OPACITY_OFFSET = numPoints * 12,
          TRANSFORM_OFFSET = numPoints * 14;
    const POINT_LOCATION = 0,
          INDEX_LOCATION = 1,
          SCALE_LOCATION = 2,
          SHAPE_LOCATION = 3,
          OPACITY_LOCATION = 4,
          TRANSFORM_LOCATION = 5;

    gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_markers"]);
    gl.vertexAttribPointer(POINT_LOCATION, 4, gl.FLOAT, false, 0, POINT_OFFSET + offset * 16);
    gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_markers_secondary"]);
    gl.vertexAttribIPointer(INDEX_LOCATION, 1, gl.INT, 0, INDEX_OFFSET + offset * 4);
    gl.vertexAttribPointer(SCALE_LOCATION, 1, gl.FLOAT, false, 0, SCALE_OFFSET + offset * 4);
    gl.vertexAttribPointer(SHAPE_LOCATION, 1, gl.FLOAT, false, 0, SHAPE_OFFSET + offset * 4);
    gl.vertexAttribPointer(OPACITY_LOCATION, 1, gl.UNSIGNED_SHORT, true, 0, OPACITY_OFFSET + offset * 2);
    gl.vertexAttribPointer(TRANSFORM_LOCATION, 1, gl.UNSIGNED_SHORT, false, 0, TRANSFORM_OFFSET + offset * 2);
}


// Create WebGL resources and other objects for drawing graph edges. Returns
// the number of edges found in the data. This function should only be called
// from within glUtils.loadMarkers().
glUtils._loadEdges = function(uid, forceUpdate) {
    if (!glUtils._initialized) return;
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl2", glUtils._options);

    let newInputs = {};  // Inputs that will require a vertex buffer update when changed

    // Get marker data and other info like image size
    const markerData = dataUtils.data[uid]["_processeddata"];
    const keyName = newInputs.keyName = dataUtils.data[uid]["_gb_col"];
    const xPosName = newInputs.xPosName = dataUtils.data[uid]["_X"];
    const yPosName = newInputs.yPosName = dataUtils.data[uid]["_Y"];
    const numPoints = markerData[xPosName].length;

    const connectionsPropertyName = newInputs.connectionsPropertyName = dataUtils.data[uid]["_edges_col"];

    // Check how the user wants to draw the edges
    glUtils._updateBarcodeToLUTIndexDict(uid, markerData, keyName);
    const barcodeToLUTIndex = glUtils._barcodeToLUTIndex[uid];
    const markerCoordFactor = newInputs.markerCoordFactor = dataUtils.data[uid]["_coord_factor"];
    const opacityPropertyName = newInputs.opacityPropertyName = dataUtils.data[uid]["_opacity_col"];
    const useOpacityFromMarker = newInputs.useOpacityFromMarker = dataUtils.data[uid]["_opacity_col"] != null;
    const markerOpacityFactor = dataUtils.data[uid]["_opacity"];
    const collectionItemPropertyName = newInputs.collectionItemPropertyName = dataUtils.data[uid]["_collectionItem_col"];
    const useCollectionItemFromMarker = newInputs.useCollectionItemFromMarker = dataUtils.data[uid]["_collectionItem_col"] != null;
    const collectionItemFixed = newInputs.collectionItemFixed = dataUtils.data[uid]["_collectionItem_fixed"];
    let collectionItemIndex = collectionItemFixed;

    // Find out how many edges there are in the data
    let numEdges = 0;
    if (markerData[connectionsPropertyName] != null) {
        for (let markerIndex = 0; markerIndex < numPoints; ++markerIndex) {
            const edges = markerData[connectionsPropertyName][markerIndex].toString().split(";");
            numEdges += edges.length;
        }
    }

    // Additional info about the vertex format. Make sure you update also
    // NUM_BYTES_PER_EDGE when making changes to the format!
    const NUM_BYTES_PER_EDGE = 24;
    const POINT_OFFSET = numEdges * 0,
          INDEX_OFFSET = numEdges * 16,
          OPACITY_OFFSET = numEdges * 20,
          TRANSFORM_OFFSET = numEdges * 22;
    const POINT_LOCATION = 0,      // Re-use the same attribute locations that
          INDEX_LOCATION = 1,      // are used also for drawing the markers
          OPACITY_LOCATION = 4,
          TRANSFORM_LOCATION = 5;

    const lastInputs = glUtils._edgeInputsCached[uid];
    if ((markerData[connectionsPropertyName] != null) &&
        (forceUpdate || (lastInputs != JSON.stringify(newInputs)))) {

        // Extract and upload vertex data for edges. Similar to for the markers, the
        // vertex data can be large, so we upload the data in chunks to the GPU
        // buffer to avoid having to allocate a large temporary buffer in system memory.
        console.time("Generate edge data");
        let chunkSize = 100000;
        let offsetEdges = 0;
        for (let offset = 0; offset < numPoints; offset += chunkSize) {
            if (offset + chunkSize >= numPoints) chunkSize = numPoints - offset;

            // Compute actual chunk size for the edges in the chunk
            let chunkSizeEdges = 0;
            for (let i = 0; i < chunkSize; ++i) {
                const markerIndex = i + offset;
                const edges = markerData[connectionsPropertyName][markerIndex].toString().split(";");
                chunkSizeEdges += edges.length;
            }

            // Allocate arrays for edge data that will be uploaded to vertex buffer
            let bytedata_point = new Float32Array(chunkSizeEdges * 4);
            let bytedata_index = new Int32Array(chunkSizeEdges * 1);
            let bytedata_opacity = new Uint16Array(chunkSizeEdges * 1);
            let bytedata_transform = new Uint16Array(chunkSizeEdges * 1);

            let offsetEdges2 = 0;
            for (let i = 0; i < chunkSize; ++i) {
                const markerIndex = i + offset;
                const lutIndex = (keyName != null) ? barcodeToLUTIndex[markerData[keyName][markerIndex]] : 0;
                const opacity = useOpacityFromMarker ? markerData[opacityPropertyName][markerIndex] : 1.0;
                if (useCollectionItemFromMarker) collectionItemIndex = markerData[collectionItemPropertyName][markerIndex];

                // Generate line segments for edges to neighboring markers
                const edges = markerData[connectionsPropertyName][markerIndex].toString().split(";");
                for (let j = 0; j < edges.length; ++j) {
                    let markerIndex_j = Number(edges[j]);
                    let lutIndex_j = (keyName != null) ? barcodeToLUTIndex[markerData[keyName][markerIndex_j]] : 0;
                    let collectionItemIndex_j = collectionItemFixed;
                    if (useCollectionItemFromMarker) collectionItemIndex_j = markerData[collectionItemPropertyName][markerIndex_j];
                    const k = offsetEdges2 + j;

                    if (markerIndex_j == 0) {
                        // FIXME Workaround for false edges to marker with index 0
                        markerIndex_j = markerIndex;
                        lutIndex_j = lutIndex;
                        collectionItemIndex_j = collectionItemIndex;
                    }

                    bytedata_point[4 * k + 0] = markerData[xPosName][markerIndex] * markerCoordFactor;
                    bytedata_point[4 * k + 1] = markerData[yPosName][markerIndex] * markerCoordFactor;
                    bytedata_point[4 * k + 2] = markerData[xPosName][markerIndex_j] * markerCoordFactor;
                    bytedata_point[4 * k + 3] = markerData[yPosName][markerIndex_j] * markerCoordFactor;
                    bytedata_index[k] = lutIndex + (lutIndex_j * 32768.0);
                    bytedata_opacity[k] = Math.floor(Math.max(0.0, Math.min(1.0, opacity)) * 65535.0);
                    bytedata_transform[k] = collectionItemIndex + (collectionItemIndex_j * 256);
                }

                offsetEdges2 += edges.length;
            }

            // Create WebGL objects (if this has not already been done)
            if (!(uid + "_edges" in glUtils._buffers))
                glUtils._buffers[uid + "_edges"] = glUtils._createVertexBuffer(gl, numEdges * NUM_BYTES_PER_EDGE);
            if (!(uid + "_edges" in glUtils._vaos))
                glUtils._vaos[uid + "_edges"] = gl.createVertexArray();

            // Upload chunks of vertex data to buffer
            gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_edges"]);
            if (offset == 0) {
                // Check if buffer needs to be allocated or re-allocated
                const newBufferSize = numEdges * NUM_BYTES_PER_EDGE;
                const oldBufferSize = gl.getBufferParameter(gl.ARRAY_BUFFER, gl.BUFFER_SIZE);
                if (newBufferSize != oldBufferSize) {
                    gl.bufferData(gl.ARRAY_BUFFER, newBufferSize, gl.STATIC_DRAW);
                }
            }

            gl.bufferSubData(gl.ARRAY_BUFFER, (POINT_OFFSET + offsetEdges * 16), bytedata_point);
            gl.bufferSubData(gl.ARRAY_BUFFER, (INDEX_OFFSET + offsetEdges * 4), bytedata_index);
            gl.bufferSubData(gl.ARRAY_BUFFER, (OPACITY_OFFSET + offsetEdges * 2), bytedata_opacity);
            gl.bufferSubData(gl.ARRAY_BUFFER, (TRANSFORM_OFFSET + offsetEdges * 2), bytedata_transform);
            gl.bindBuffer(gl.ARRAY_BUFFER, null);
            offsetEdges += chunkSizeEdges;
        }
        console.timeEnd("Generate edge data");

        // Set up VAO with vertex format for drawing thick lines via instancing
        gl.bindVertexArray(glUtils._vaos[uid + "_edges"]);
        gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_edges"]);
        gl.enableVertexAttribArray(POINT_LOCATION);
        gl.vertexAttribPointer(POINT_LOCATION, 4, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(POINT_LOCATION, 1);
        gl.enableVertexAttribArray(INDEX_LOCATION);
        gl.vertexAttribIPointer(INDEX_LOCATION, 1, gl.INT, 0, INDEX_OFFSET);
        gl.vertexAttribDivisor(INDEX_LOCATION, 1);
        gl.enableVertexAttribArray(OPACITY_LOCATION);
        gl.vertexAttribPointer(OPACITY_LOCATION, 1, gl.UNSIGNED_SHORT, true, 0, OPACITY_OFFSET);
        gl.vertexAttribDivisor(OPACITY_LOCATION, 1);
        gl.enableVertexAttribArray(TRANSFORM_LOCATION);
        gl.vertexAttribPointer(TRANSFORM_LOCATION, 1, gl.UNSIGNED_SHORT, false, 0, TRANSFORM_OFFSET);
        gl.vertexAttribDivisor(TRANSFORM_LOCATION, 1);
        gl.bindVertexArray(null);
    }
    glUtils._edgeInputsCached[uid] = JSON.stringify(newInputs);

    return numEdges;
}


glUtils._updateBindingOffsetsForCurrentEdgesVAO = function(gl, uid, offset, numEdges) {
    // This function is used for updating the offsets of vertex bindings for
    // instanced drawing. This is only necessary because drawArrayInstanced and
    // drawElementsInstanced ignores any offset/first parameters we provide
    // for instanced arrays (arrays with non-zero vertex attrib divisor).

    // Additional info about the vertex format. Make sure to update these values
    // if/when also changing the vertex format in glUtils.loadEdges!    
    const POINT_OFFSET = numEdges * 0,
          INDEX_OFFSET = numEdges * 16,
          OPACITY_OFFSET = numEdges * 20,
          TRANSFORM_OFFSET = numEdges * 22;
    const POINT_LOCATION = 0,
          INDEX_LOCATION = 1,
          OPACITY_LOCATION = 4,
          TRANSFORM_LOCATION = 5;

    gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers[uid + "_edges"]);
    gl.vertexAttribPointer(POINT_LOCATION, 4, gl.FLOAT, false, 0, POINT_OFFSET + offset * 16);
    gl.vertexAttribIPointer(INDEX_LOCATION, 1, gl.INT, 0, INDEX_OFFSET + offset * 4);
    gl.vertexAttribPointer(OPACITY_LOCATION, 1, gl.UNSIGNED_SHORT, true, 0, OPACITY_OFFSET + offset * 2);
    gl.vertexAttribPointer(TRANSFORM_LOCATION, 1, gl.UNSIGNED_SHORT, false, 0, TRANSFORM_OFFSET + offset * 2);
}


glUtils._updateBarcodeToLUTIndexDict = function (uid, markerData, keyName, maxNumBarcodes=32768) {
    const barcodeToLUTIndex = {};
    const barcodeToKey = {};
    const numPoints = markerData[markerData.columns[0]].length;
    console.log("Key name: " + keyName);
    for (let i = 0, index = 0; i < numPoints; ++i) {
        const barcode = (keyName != null) ? markerData[keyName][i] : undefined;
        if (!(barcode in barcodeToLUTIndex)) {
            barcodeToLUTIndex[barcode] = index++;
            barcodeToKey[barcode] = barcode;
            index = index % maxNumBarcodes;  // Prevent index from becoming >= the maximum LUT size,
                                             // since this causes problems with pie-chart markers
        }
    }
    if (Object.keys(barcodeToLUTIndex).length > maxNumBarcodes) {
        interfaceUtils.generateNotification("Markerset contains more than " + maxNumBarcodes +
            " number of unique keys or groups. Some parts of the visualization " +
            "(colors and selection) might not work correctly.", "barcode_warning", false, false);
    }
    glUtils._barcodeToLUTIndex[uid] = barcodeToLUTIndex;
    glUtils._barcodeToKey[uid] = barcodeToKey;
    console.log("barcodeToLUTIndex, barcodeToKey", barcodeToLUTIndex, barcodeToKey);
}


glUtils._createColorLUTTexture = function(gl, maxNumBarcodes=32768) {
    console.assert((maxNumBarcodes % 4096) == 0);  // Must be a multiple of the LUT texture width

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, 4096, maxNumBarcodes / 4096);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return texture;
}


glUtils._updateColorLUTTexture = function(gl, uid, texture, maxNumBarcodes=32768) {
    console.assert((maxNumBarcodes % 4096) == 0);  // Must be a multiple of the LUT texture width
    if (!(uid + "_colorLUT" in glUtils._textures)) return;

    const hasGroups = dataUtils.data[uid]["_gb_col"] != null;

    const colors = new Array(maxNumBarcodes * 4);
    for (let [barcode, index] of Object.entries(glUtils._barcodeToLUTIndex[uid])) {
        const key = hasGroups ? glUtils._barcodeToKey[uid][barcode] : "All";
        const inputs = interfaceUtils._mGenUIFuncs.getGroupInputs(uid, key);
        const hexColor = "color" in inputs ? inputs["color"] : "#ffff00";
        const shape = "shape" in inputs ? inputs["shape"] : "circle";
        const visible = "visible" in inputs ? inputs["visible"] : true;
        const hidden = "hidden" in inputs ? inputs["hidden"] : true;
        // OBS! Need to clamp this value, since indexOf() can return -1
        const shapeIndex = Math.max(0, markerUtils._symbolStrings.indexOf(shape));

        colors[4 * index + 0] = Number("0x" + hexColor.substring(1,3)); 
        colors[4 * index + 1] = Number("0x" + hexColor.substring(3,5));
        colors[4 * index + 2] = Number("0x" + hexColor.substring(5,7));
        colors[4 * index + 3] = Number(visible) * (1 - Number(hidden)) * (Number(shapeIndex) + 1);
    }

    const bytedata = new Uint8Array(colors);

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 4096, maxNumBarcodes / 4096,
                     gl.RGBA, gl.UNSIGNED_BYTE, bytedata);
    gl.bindTexture(gl.TEXTURE_2D, null);
}


/**
 * @summary Update the color scale LUTs for all marker datasets.
 * This function is a callback and should not normally be called directly.
 */
glUtils.updateColorLUTTextures = function() {
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl2", glUtils._options);

    for (let [uid, numPoints] of Object.entries(glUtils._numPoints)) {
        glUtils._updateColorLUTTexture(gl, uid, glUtils._textures[uid + "_colorLUT"]);
    }
}


glUtils._updateTransformUBO = function(buffer) {
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl2", glUtils._options);

    // Compute transforms that takes into account if collection mode viewing is
    // enabled for image layers
    const imageTransforms = new Array(256 * 8).fill(0);
    for (let i = 0; i < tmapp["ISS_viewer"].world.getItemCount(); ++i) {
        const bounds = tmapp["ISS_viewer"].viewport.getBounds();
        const image = tmapp["ISS_viewer"].world.getItemAt(i);
        const imageWidth = image.getContentSize().x;
        const imageHeight = image.getContentSize().y;
        const imageBounds = image.getBounds();
        const imageBounds2 = image.getBoundsNoRotate();
        const imageFlip = image.getFlip();
        let imageOrientation = image.getRotation();
        // Make sure rotation angle is in the range [0, 360) degrees
        imageOrientation = imageOrientation - Math.floor(imageOrientation / 360.0) * 360.0;

        const theta = imageOrientation * (3.14159265 / 180.0);
        const flip = imageFlip ? -1.0 : 1.0;
        const scaleX = (imageBounds2.width / imageWidth) / bounds.width;
        const scaleY = (imageBounds2.height / imageHeight) / bounds.height;
        const shiftX = -(bounds.x - imageBounds.x) / bounds.width;
        const shiftY = -(bounds.y - imageBounds.y) / bounds.height;
        // HACK: Extra constants for compensating for how OpenSeaDragon handles rotation.
        const k0 = (imageOrientation >= 180.0) ? (imageFlip ? 0.0 : 1.0) : (imageFlip ? -1.0 : 0.0);
        const k1 = (imageOrientation >= 90.0 && imageOrientation < 270.0) ? 1.0 : 0.0;

        // Construct 3x2 matrix for transform. The matrix is stored transposed
        // in a 2x4 matrix with column-major order, to comply with std140
        // alignment rules for storing matrices in arrays in UBOs.
        imageTransforms[i * 8 + 0] = flip * Math.cos(theta) * scaleX;
        imageTransforms[i * 8 + 1] = -Math.sin(theta) * scaleX;
        imageTransforms[i * 8 + 2] = shiftX - k0 * (scaleX * imageWidth) * Math.cos(theta) + k1 * (scaleX * imageHeight) * Math.sin(theta);
        imageTransforms[i * 8 + 4] = flip * Math.sin(theta) * scaleY;
        imageTransforms[i * 8 + 5] = Math.cos(theta) * scaleY;
        imageTransforms[i * 8 + 6] = shiftY - k0 * (scaleY * imageWidth) * Math.sin(theta) - k1 * (scaleY * imageHeight) * Math.cos(theta);
    }

    const bytedata = new Float32Array(imageTransforms);

    gl.bindBuffer(gl.UNIFORM_BUFFER, buffer);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, bytedata);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
}


/**
 * @summary Update the region data textures for all region datasets.
 * This function is a callback and should not normally be called directly.
 */
glUtils.updateRegionDataTextures = function() {
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl2", glUtils._options);

    console.time("Update region edge lists");
    regionUtils._generateEdgeListsForDrawing();
    console.timeEnd("Update region edge lists");

    // console.time("Split region edge lists");
    // regionUtils._splitEdgeLists();
    // console.timeEnd("Split region edge lists");

    console.time("Add clusters to region edge lists");
    regionUtils._addClustersToEdgeLists();
    console.timeEnd("Add clusters to region edge lists");

    for (let collectionIndex in regionUtils._edgeListsByLayer) {
        const textureCreateInfo = [
            {name: "regionData_" + collectionIndex, useSplit: false},
            // {name: "regionDataSplit_" + collectionIndex, useSplit: true}
        ];

        for (let item of textureCreateInfo) {
            // Compute size required for storing texture data
            const numTexels =
                glUtils._updateRegionDataTexture(gl, null, collectionIndex, item.useSplit);

            // Check if texture needs to be created or resized
            if (!(item.name in glUtils._textures)) {
                glUtils._textures[item.name] = glUtils._createRegionDataTexture(gl, numTexels);
            } else if (glUtils._regionDataSize[item.name] < numTexels) {
                gl.deleteTexture(glUtils._textures[item.name]);
                glUtils._textures[item.name] = glUtils._createRegionDataTexture(gl, numTexels);
            }
            glUtils._regionDataSize[item.name] = numTexels;

            glUtils._updateRegionDataTexture(
                gl, glUtils._textures[item.name], collectionIndex, item.useSplit);
        }
    }
}


glUtils._createRegionDataTexture = function(gl, numTexels) {
    console.assert(numTexels > 0);
    console.assert((numTexels % 4096) == 0);

    const height = numTexels / 4096;
    // Clamp height to reported maximum texture size to be safe
    const heightAdjusted = Math.min(height, glUtils._caps[gl.MAX_TEXTURE_SIZE]);

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); 
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 4096, heightAdjusted);
    // Do explicit initialization with zeros, to avoid Firefox warning about
    // "Tex image TEXTURE_2D level 0 is incurring lazy initialization."
    // when texSubImage2D() is used to only partially update the texture
    let zeros = new Float32Array(4096 * heightAdjusted * 4);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 4096, heightAdjusted, gl.RGBA, gl.FLOAT, zeros);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return texture;
}


glUtils._updateRegionDataTexture = function(gl, texture, collectionIndex=0, useSplitData=false) {
    if (!(collectionIndex in regionUtils._edgeListsByLayer)) return 0;

    const edgeLists = useSplitData ? regionUtils._edgeListsByLayerSplit[collectionIndex]
                                   : regionUtils._edgeListsByLayerClustered[collectionIndex];
    const numScanlines = edgeLists.length;
    const numSides = useSplitData ? 2 : 1;

    // The region data texture will require space for storing scanline pointers
    // and lengths in the first (numScanlines * 2) texels, followed by space for
    // the scanline data (edges and bounding boxes) tightly packed in the rest
    // of the texture.

    let numTexels = numScanlines * 2;
    for (let j = 0; j < numSides; ++j) {
        for (let i = 0; i < numScanlines; ++i) {
            const count = edgeLists[i][j].length / 4;
            numTexels += count + 1;  // Include zero texel for indicating end of scanline
        }
    }
    numTexels += 4096 - (numTexels % 4096);  // Pad to multiple of texture width

    if (texture != null) {
        let texeldata = new Float32Array(numTexels * 4);  // Zero-initialized
        let offset = numScanlines * 2;
        for (let j = 0; j < numSides; ++j) {
            for (let i = 0; i < numScanlines; ++i) {
                const count = edgeLists[i][j].length / 4;
                texeldata.set(edgeLists[i][j], offset * 4);
                // Storing the scanline offset as a float only allows exact
                // representation of integers up to 2^24, so need to split it
                // into two X and Y offsets instead
                texeldata[4 * (i + numScanlines * j) + 0] = offset % 4096;
                texeldata[4 * (i + numScanlines * j) + 1] = Math.floor(offset / 4096);
                offset += count + 1;  // Include zero texel for indicating end of scanline
            }
        }

        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 4096, numTexels / 4096,
                         gl.RGBA, gl.FLOAT, texeldata);
        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    // Return size used for creating (or resizing) the texture (if no texture
    // handle was passed as input to this function)
    return numTexels;
}


/**
 * @summary Update the color/visibility LUTs for all region datasets.
 * This function is a callback and should not normally be called directly.
 */
glUtils.updateRegionLUTTextures = function() {
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl2", glUtils._options);

    console.time("Update region LUT");
    regionUtils._generateRegionToColorLUT();
    console.timeEnd("Update region LUT");

    const numRegions = regionUtils._regionToColorLUT.length / 4;
    if (numRegions > glUtils._regionMaxNumRegions) {
        gl.deleteTexture(glUtils._textures["regionLUT"]);

        // Increase maximum LUT size to closest power-of-two greater than or equal to numRegions
        glUtils._regionMaxNumRegions = (1 << Math.ceil(Math.log2(numRegions)));
        glUtils._textures["regionLUT"] =
            glUtils._createRegionLUTTexture(gl, glUtils._regionMaxNumRegions);
    }
    console.assert(numRegions <= glUtils._regionMaxNumRegions);
    glUtils._updateRegionLUTTexture(gl, glUtils._textures["regionLUT"], glUtils._regionMaxNumRegions);
}


glUtils._createRegionLUTTexture = function(gl, maxNumRegions) {
    console.assert((maxNumRegions % 4096) == 0);  // Must be a multiple of the LUT texture width

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, 4096, maxNumRegions / 4096);
    // Do explicit initialization with zeros, to avoid Firefox warning about
    // "Tex image TEXTURE_2D level 0 is incurring lazy initialization."
    // when texSubImage2D() is used to only partially update the texture
    let zeros = new Uint8Array(maxNumRegions * 4);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 4096, maxNumRegions / 4096,
                     gl.RGBA, gl.UNSIGNED_BYTE, zeros);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return texture;
}


glUtils._updateRegionLUTTexture = function(gl, texture, maxNumRegions) {
    console.assert((maxNumRegions % 4096) == 0);  // Must be a multiple of the LUT texture width

    let texeldata = new Uint8Array(maxNumRegions * 4);
    console.assert(regionUtils._regionToColorLUT.length <= texeldata.length);
    texeldata.set(regionUtils._regionToColorLUT);

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 4096, maxNumRegions / 4096,
                     gl.RGBA, gl.UNSIGNED_BYTE, texeldata);
    gl.bindTexture(gl.TEXTURE_2D, null);
}


glUtils._createColorScaleTexture = function(gl) {
    const bytedata = new Uint8Array(256 * 4);

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA,
                  gl.UNSIGNED_BYTE, bytedata);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return texture;
}


glUtils._formatHex = function(color) {
    if (color.includes("rgb")) {
        const r = color.split(",")[0].replace("rgb(", "").replace(")", "");
        const g = color.split(",")[1].replace("rgb(", "").replace(")", "");
        const b = color.split(",")[2].replace("rgb(", "").replace(")", "");
        const hex = (Number(r) * 65536 + Number(g) * 256 + Number(b)).toString(16);
        color = "#" + ("0").repeat(6 - hex.length) + hex;
    }
    return color;
}


glUtils._updateColorScaleTexture = function(gl, uid, texture) {
    const colors = [];
    const colorscaleName = glUtils._colorscaleName[uid];
    console.log(colorscaleName);
    for (let i = 0; i < 256; ++i) {
        const normalized = i / 255.0;
        if (colorscaleName.includes("interpolate") &&
            !colorscaleName.includes("Rainbow")) {
            const color = d3[colorscaleName](normalized);
            const hexColor = glUtils._formatHex(color);  // D3 sometimes returns RGB strings
            colors[4 * i + 0] = Number("0x" + hexColor.substring(1,3));
            colors[4 * i + 1] = Number("0x" + hexColor.substring(3,5));
            colors[4 * i + 2] = Number("0x" + hexColor.substring(5,7));
            colors[4 * i + 3] = 255.0;
        } else {
            // Use a version of Google's Turbo colormap with brighter blue range
            // Reference: https://www.shadertoy.com/view/WtGBDw
            const r = Math.sin((normalized - 0.33) * 3.141592);
            const g = Math.sin((normalized + 0.00) * 3.141592);
            const b = Math.sin((normalized + 0.33) * 3.141592);
            const s = 1.0 - normalized;  // For purplish tone at end of the range
            colors[4 * i + 0] = Math.max(0.0, Math.min(1.0, r * (1.0 - 0.5 * b*b) + s*s)) * 255.99;
            colors[4 * i + 1] = Math.max(0.0, Math.min(1.0, g * (1.0 - r*r * b*b))) * 255.99;
            colors[4 * i + 2] = Math.max(0.0, Math.min(1.0, b * (1.0 - 0.5 * r*r))) * 255.99;
            colors[4 * i + 3] = 255.0;
        }
    }
    glUtils._colorscaleData[uid] = colors;

    const bytedata = new Uint8Array(colors);

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA,
                  gl.UNSIGNED_BYTE, bytedata);
    gl.bindTexture(gl.TEXTURE_2D, null);
}


glUtils._updateColorbarCanvas = function(resolution) {
    if (resolution == undefined) resolution = 1.;
    const canvas = document.getElementById("colorbar_canvas");
    const ctx = canvas.getContext("2d");

    // Determine canvas height needed to show colorbars for all markersets that
    // have colormaps
    let canvasHeight = 0;
    const rowHeight = 70;  // Note: hardcoded value
    for (let [uid, numPoints] of Object.entries(glUtils._numPoints)) {
        if (glUtils._showColorbar && glUtils._useColorFromColormap[uid])
            canvasHeight += rowHeight + 10;
    }
    canvasHeight -= 10; // No margin for last colorbar 

    // Resize and clear canvas
    ctx.canvas.height = canvasHeight * resolution;
    ctx.canvas.width = 266 * resolution;
    ctx.canvas.style.marginTop = -canvasHeight + "px";
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    if (canvasHeight == -10) {
        ctx.canvas.className = "d-none";
        return;  // Nothing more to do for empty canvas
    }
    ctx.canvas.className = "viewer-layer";
    // Create colorbars for the markersets
    let yOffset = 0;
    for (let [uid, numPoints] of Object.entries(glUtils._numPoints)) {
        if (!glUtils._useColorFromColormap[uid]) continue;

        const propertyRange = glUtils._markerScalarRange[uid];
        const propertyName = glUtils._markerScalarPropertyName[uid];
        const colorscaleData = glUtils._colorscaleData[uid];

        // Define gradient for color scale
        const gradient = ctx.createLinearGradient(5* resolution, 0, (256+5)* resolution, 0);
        const numStops = 32;
        for (let i = 0; i < numStops; ++i) {
            const normalized = i / (numStops - 1);
            const index = Math.floor(normalized * 255.99);
            const r = Math.floor(colorscaleData[4 * index + 0]);
            const g = Math.floor(colorscaleData[4 * index + 1]);
            const b = Math.floor(colorscaleData[4 * index + 2]);
            gradient.addColorStop(Math.min(1,normalized
                ), "rgb(" + r + "," + g + "," + b + ")");
        }
        // Draw colorbar (with outline)
        ctx.fillStyle = gradient;
        ctx.fillRect(5 * resolution, (48 + yOffset) * resolution, 256 * resolution, 16 * resolution);
        ctx.strokeStyle = "#555";
        ctx.strokeRect(5 * resolution, (48 + yOffset) * resolution, 256 * resolution, 16 * resolution);

        // Convert range annotations to precision 7 and remove trailing zeros
        let propertyMin = propertyRange[0].toPrecision(7).replace(/\.([^0]+)0+$/,".$1");
        let propertyMax = propertyRange[1].toPrecision(7).replace(/\.([^0]+)0+$/,".$1");
        // Convert range annotations to scientific notation if they may overflow
        if (propertyMin.length > 9) propertyMin = propertyRange[0].toExponential(5);
        if (propertyMax.length > 9) propertyMax = propertyRange[1].toExponential(5);
        // Get marker tab name to show together with property name
        const tabName = interfaceUtils.getElementById(uid + "_marker-tab-name").textContent;
        // let label = tabName.substring(0, 15) + "." + propertyName.substring(0, 15);
        let label = tabName.substring(0, 30);
        // Draw annotations (with drop shadow)
        ctx.font = (16 * resolution) + "px Segoe UI";
        ctx.textAlign = "center";
        ctx.fillStyle = "#000";  // Shadow color
        ctx.fillText(label, (ctx.canvas.width/2+1), (18+1 + yOffset) * resolution);
        ctx.textAlign = "left";
        ctx.fillText(propertyMin, (ctx.canvas.width/2-128*resolution+1), (40+1 + yOffset) * resolution);
        ctx.textAlign = "right";
        ctx.fillText(propertyMax, (ctx.canvas.width/2+128*resolution+1), (40+1 + yOffset) * resolution);
        yOffset += rowHeight + 10;  // Move to next colorbar row
    }
}


// Creates a 2D-canvas for drawing the colorbar on top of the WebGL-canvas
glUtils._createColorbarCanvas = function() {
    const root = document.getElementById("gl_canvas").parentElement;
    const canvas = document.createElement("canvas");
    root.appendChild(canvas);

    canvas.id = "colorbar_canvas";
    canvas.className = "d-none";
    canvas.width = "266";  // Fixed width in pixels
    canvas.height = "96";  // Fixed height in pixels
    canvas.style = "position:relative; float:right; width:266px; bottom: 11px; right: 14px; " +
                   "margin-top:-96px; z-index:20; pointer-events:none";
}


glUtils._createQuad = function(gl) {
    // Create geometry for drawing a single quad
    const vertices = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    const bytedata = new Float32Array(vertices);

    const buffer = glUtils._createVertexBuffer(gl, bytedata.byteLength);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, bytedata);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    return buffer;
}


// Creates WebGL canvas for drawing the markers
glUtils._createMarkerWebGLCanvas = function() {
    const canvas = document.createElement("canvas");
    canvas.id = "gl_canvas";
    canvas.width = "1"; canvas.height = "1";
    canvas.style = "position:relative; pointer-events:none; z-index: 12; width: 100%; height: 100%";
    return canvas;
}


glUtils._loadTextureFromImageURL = function(gl, src) {
    const texture = gl.createTexture();
    const image = new Image();
    image.onload = function() {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR); 
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
        gl.generateMipmap(gl.TEXTURE_2D);  // Requires power-of-two size images
        gl.bindTexture(gl.TEXTURE_2D, null);
        glUtils.draw();  // Force redraw to avoid black shapes after context loss
    };
    image.src = src;
    return texture;
}


glUtils._drawColorPass = function(gl, viewportTransform, markerScaleAdjusted) {
    let drawOrder = [];
    for (let [uid, _] of Object.entries(glUtils._numPoints)) { drawOrder.push(uid); }
    // Sort draws in ascending z-order (Note: want stable sort so that
    // we retain the old behaviour when z-order is the same for all UIDs)
    drawOrder.sort((a, b) => glUtils._zOrder[a] - glUtils._zOrder[b]);

    if (!glUtils._regionShowOnTop) {
        // Draw regions first, so that they appear under markers
        glUtils._drawRegionsColorPass(gl, viewportTransform);
    }

    // Draw markers and edges interleaved, to ensure correct overlap per UID
    for (let uid of drawOrder) {
        if (glUtils._showEdgesExperimental) {
            glUtils._drawEdgesByUID(gl, viewportTransform, markerScaleAdjusted, uid);
        }
        glUtils._drawMarkersByUID(gl, viewportTransform, markerScaleAdjusted, uid);
    }

    if (glUtils._regionShowOnTop) {
        // Draw regions last, so that they appear on top of markers (this is the
        // same behaviour as for the old SVG regions)
        glUtils._drawRegionsColorPass(gl, viewportTransform);
    }
}


glUtils._drawMarkersByUID = function(gl, viewportTransform, markerScaleAdjusted, uid) {
    const numPoints = glUtils._numPoints[uid];
    if (numPoints == 0) return;

    // Chunk size used to split a single large draw call into smaller chunks. On
    // some Android phones, drawing larger datasets can result in WebGL context
    // loss or crashes, so this should be a workaround.
    const chunkSize = 65536;

    // Set up render pipeline
    const program = glUtils._programs[glUtils._useInstancing ? "markers_instanced" : "markers"];
    gl.useProgram(program);
    gl.enable(gl.BLEND);
    if (glUtils._markerBlendMode[uid] == "coverage") {
        // Note: this blend mode will invert the draw order, s.t. that if
        // markers for example were sorted by scalar value in ascending order,
        // the marker with the lowest scalar value will appear on top! A
        // workaround is to just reverse the sorting when using this mode.
        gl.blendFuncSeparate(gl.SRC_ALPHA_SATURATE, gl.ONE, gl.ONE, gl.ONE);
    } else if (glUtils._markerBlendMode[uid] == "add") {
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE, gl.ONE, gl.ONE);
    } else if (glUtils._markerBlendMode[uid] == "max") {
        // This looks quite bad when premultiplied alpha is used to composite
        // the WebGL canvas. So might want to manually set premultipliedAlpha to
        // false via glUtils._options in projects that need this blend mode.
        gl.blendEquationSeparate(gl.MAX, gl.MAX);
    } else {  // glUtils._markerBlendMode[uid] == "over"
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    }

    // Set per-scene uniforms
    gl.uniformMatrix2fv(gl.getUniformLocation(program, "u_viewportTransform"), false, viewportTransform);
    gl.uniform2fv(gl.getUniformLocation(program, "u_canvasSize"), [gl.canvas.width, gl.canvas.height]);
    gl.uniform1f(gl.getUniformLocation(program, "u_globalMarkerScale"), glUtils._globalMarkerScale * markerScaleAdjusted);
    gl.uniform1f(gl.getUniformLocation(program, "u_maxPointSize"),
        glUtils._useInstancing ? 2048 : glUtils._caps[gl.ALIASED_POINT_SIZE_RANGE][1]);
    gl.uniformBlockBinding(program, gl.getUniformBlockIndex(program, "TransformUniforms"), 0);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, glUtils._buffers["transformUBO"]);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, glUtils._textures["shapeAtlas"]);
    gl.uniform1i(gl.getUniformLocation(program, "u_shapeAtlas"), 2);

    gl.bindVertexArray(glUtils._vaos[uid + (glUtils._useInstancing ? "_markers_instanced" : "_markers")]);

    // Set per-markerset uniforms
    gl.uniform1i(gl.getUniformLocation(program, "u_transformIndex"),
        glUtils._collectionItemIndex[uid] != null ? glUtils._collectionItemIndex[uid] : -1);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale"), (glUtils._useAbsoluteMarkerSize[uid] ? glUtils._markerScale * 0.25 : 1.0) * glUtils._markerScaleFactor[uid]);
    gl.uniform2fv(gl.getUniformLocation(program, "u_markerScalarRange"), glUtils._markerScalarRange[uid]);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerOpacity"), glUtils._markerOpacity[uid]);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerStrokeWidth"), glUtils._markerStrokeWidth[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_markerFilled"), glUtils._markerFilled[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_markerOutline"), glUtils._markerOutline[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_useColorFromMarker"), glUtils._useColorFromMarker[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_useColorFromColormap"), glUtils._useColorFromColormap[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_usePiechartFromMarker"), glUtils._usePiechartFromMarker[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_useShapeFromMarker"), glUtils._useShapeFromMarker[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_useAbsoluteMarkerSize"), glUtils._useAbsoluteMarkerSize[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_pickedMarker"),
        glUtils._pickedMarker[0] == uid ? glUtils._pickedMarker[1] : -1);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, glUtils._textures[uid + "_colorscale"]);
    gl.uniform1i(gl.getUniformLocation(program, "u_colorscale"), 1);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, glUtils._textures[uid + "_colorLUT"]);
    gl.uniform1i(gl.getUniformLocation(program, "u_colorLUT"), 0);

    if (glUtils._usePiechartFromMarker[uid]) {
        // 1st pass: draw alpha for whole marker shapes
        gl.uniform1i(gl.getUniformLocation(program, "u_alphaPass"), true);
        for (let offset = 0; offset < numPoints; offset += chunkSize) {
            const count = (offset + chunkSize >= numPoints) ? numPoints - offset : chunkSize;
            if (glUtils._useInstancing) {
                glUtils._updateBindingOffsetsForCurrentMarkerVAO(gl, uid, offset, numPoints);
                gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
            } else {
                gl.drawArrays(gl.POINTS, offset, count);
            }
        }
        // 2nd pass: draw colors for individual piechart sectors
        gl.uniform1i(gl.getUniformLocation(program, "u_alphaPass"), false);
        gl.colorMask(true, true, true, false);
        // (Reminder of the drawing is the same as for non-piechart markers, so
        // here we can just re-use the code that follows)
    }
    for (let offset = 0; offset < numPoints; offset += chunkSize) {
        const count = (offset + chunkSize >= numPoints) ? numPoints - offset : chunkSize;
        if (glUtils._useInstancing) {
            glUtils._updateBindingOffsetsForCurrentMarkerVAO(gl, uid, offset, numPoints);
            gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
        } else {
            gl.drawArrays(gl.POINTS, offset, count);
        }
    }

    // Restore render pipeline state
    if (glUtils._useInstancing) {
        glUtils._updateBindingOffsetsForCurrentMarkerVAO(gl, uid, 0, numPoints);
    }
    gl.bindVertexArray(null);
    gl.colorMask(true, true, true, true);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.blendEquation(gl.FUNC_ADD);
    gl.disable(gl.BLEND);
    gl.useProgram(null);
}


glUtils._drawEdgesByUID = function(gl, viewportTransform, markerScaleAdjusted, uid) {
    const numEdges = glUtils._numEdges[uid];
    if (numEdges == 0) return;

    // Chunk size used to split a single large draw call into smaller chunks. On
    // some Android phones, drawing larger datasets can result in WebGL context
    // loss or crashes, so this should be a workaround.
    const chunkSize = 65536;

    // Set up render pipeline
    const program = glUtils._programs["edges"];
    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    // Set per-scene uniforms
    gl.uniformMatrix2fv(gl.getUniformLocation(program, "u_viewportTransform"), false, viewportTransform);
    gl.uniform2fv(gl.getUniformLocation(program, "u_canvasSize"), [gl.canvas.width, gl.canvas.height]);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale"), markerScaleAdjusted);
    gl.uniform1f(gl.getUniformLocation(program, "u_maxPointSize"),
        glUtils._useInstancing ? 2048 : glUtils._caps[gl.ALIASED_POINT_SIZE_RANGE][1]);
    gl.uniform1f(gl.getUniformLocation(program, "u_edgeThicknessRatio"), glUtils._edgeThicknessRatio);
    gl.uniformBlockBinding(program, gl.getUniformBlockIndex(program, "TransformUniforms"), 0);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, glUtils._buffers["transformUBO"]);

    // Set per-markerset uniforms
    gl.uniform1f(gl.getUniformLocation(program, "u_globalMarkerScale"), glUtils._globalMarkerScale * glUtils._markerScaleFactor[uid]);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerOpacity"), glUtils._markerOpacity[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_transformIndex"),
        glUtils._collectionItemIndex[uid] != null ? glUtils._collectionItemIndex[uid] : -1);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, glUtils._textures[uid + "_colorLUT"]);
    gl.uniform1i(gl.getUniformLocation(program, "u_colorLUT"), 0);

    gl.bindVertexArray(glUtils._vaos[uid + "_edges"]);
    for (let offset = 0; offset < numEdges; offset += chunkSize) {
        const count = (offset + chunkSize >= numEdges) ? numEdges - offset : chunkSize;
        glUtils._updateBindingOffsetsForCurrentEdgesVAO(gl, uid, offset, numEdges);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
    }

    // Restore render pipeline state
    gl.bindVertexArray(null);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.disable(gl.BLEND);
    gl.useProgram(null);
}


glUtils._drawRegionsColorPass = function(gl, viewportTransform) {
    if (Object.keys(regionUtils._edgeListsByLayer).length == 0) return;  // No regions to draw

    const fillRuleConstants = { "never": 0, "nonzero": 1, "oddeven": 2 };

    // Set up render pipeline
    const program = glUtils._programs["regions"];
    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    for (let collectionIndex in regionUtils._edgeListsByLayer) {
        const edgeLists = regionUtils._edgeListsByLayer[collectionIndex];
        const numScanlines = edgeLists.length;

        const image = tmapp["ISS_viewer"].world.getItemAt(collectionIndex);
        console.assert(image != undefined);
        const imageWidth = image.getContentSize().x;
        const imageHeight = image.getContentSize().y;
        const imageBounds = [0, 0, imageWidth, imageHeight];

        // Set per-scene uniforms
        gl.uniformMatrix2fv(gl.getUniformLocation(program, "u_viewportTransform"), false, viewportTransform);
        gl.uniform2fv(gl.getUniformLocation(program, "u_canvasSize"), [gl.canvas.width, gl.canvas.height]);
        gl.uniform1i(gl.getUniformLocation(program, "u_transformIndex"), collectionIndex);
        gl.uniform4fv(gl.getUniformLocation(program, "u_imageBounds"), imageBounds);
        gl.uniform1i(gl.getUniformLocation(program, "u_numScanlines"), numScanlines);
        gl.uniform1f(gl.getUniformLocation(program, "u_regionOpacity"), glUtils._regionOpacity);
        gl.uniform1f(gl.getUniformLocation(program, "u_regionStrokeWidth"), glUtils._regionStrokeWidth);
        gl.uniform1i(gl.getUniformLocation(program, "u_regionFillRule"),
            fillRuleConstants[glUtils._regionFillRule]);
        gl.uniform1i(gl.getUniformLocation(program, "u_regionUsePivotSplit"), glUtils._regionUsePivotSplit);
        gl.uniform1i(gl.getUniformLocation(program, "u_regionUseColorByID"), glUtils._regionUseColorByID);
        gl.uniformBlockBinding(program, gl.getUniformBlockIndex(program, "TransformUniforms"), 0);
        gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, glUtils._buffers["transformUBO"]);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, glUtils._textures["regionLUT"]);
        gl.uniform1i(gl.getUniformLocation(program, "u_regionLUT"), 2);
        gl.activeTexture(gl.TEXTURE1);
        if (glUtils._regionUsePivotSplit) {
            gl.bindTexture(gl.TEXTURE_2D, glUtils._textures["regionDataSplit_" + collectionIndex]);
        } else {
            gl.bindTexture(gl.TEXTURE_2D, glUtils._textures["regionData_" + collectionIndex]);
        }
        gl.uniform1i(gl.getUniformLocation(program, "u_regionData"), 1);

        // Draw rectangle that will render the regions in the fragment shader
        gl.bindVertexArray(glUtils._vaos["empty"]);
        gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers["quad"]);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, 1);
        gl.disableVertexAttribArray(0);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    // Restore render pipeline state
    gl.bindVertexArray(null);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.disable(gl.BLEND);
    gl.useProgram(null);
}


glUtils._drawPickingPass = function(gl, viewportTransform, markerScaleAdjusted) {
    // Chunk size used to split a single large draw call into smaller chunks. On
    // some Android phones, drawing larger datasets can result in WebGL context
    // loss or crashes, so this should be a workaround.
    const chunkSize = 65536;

    // Set up render pipeline
    const program = glUtils._programs["picking"];
    gl.useProgram(program);

    // Set per-scene uniforms
    gl.uniformMatrix2fv(gl.getUniformLocation(program, "u_viewportTransform"), false, viewportTransform);
    gl.uniform2fv(gl.getUniformLocation(program, "u_canvasSize"), [gl.canvas.width, gl.canvas.height]);
    gl.uniform2fv(gl.getUniformLocation(program, "u_pickingLocation"), glUtils._pickingLocation);
    gl.uniform1f(gl.getUniformLocation(program, "u_globalMarkerScale"), glUtils._globalMarkerScale * markerScaleAdjusted);
    gl.uniform1f(gl.getUniformLocation(program, "u_maxPointSize"),
        glUtils._useInstancing ? 2048 : glUtils._caps[gl.ALIASED_POINT_SIZE_RANGE][1]);
    gl.uniformBlockBinding(program, gl.getUniformBlockIndex(program, "TransformUniforms"), 0);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, glUtils._buffers["transformUBO"]);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, glUtils._textures["shapeAtlas"]);
    gl.uniform1i(gl.getUniformLocation(program, "u_shapeAtlas"), 2);

    glUtils._pickedMarker = [-1, -1];  // Reset to no picked marker

    let drawOrder = [];
    for (let [uid, _] of Object.entries(glUtils._numPoints)) { drawOrder.push(uid); }
    // Sort draws in ascending z-order (Note: want stable sort so that
    // we retain the old behaviour when z-order is the same for all UIDs)
    drawOrder.sort((a, b) => glUtils._zOrder[a] - glUtils._zOrder[b]);

    for (let uid of drawOrder) {
        const numPoints = glUtils._numPoints[uid];
        if (numPoints == 0) continue;

        gl.bindVertexArray(glUtils._vaos[uid + "_markers"]);

        // Set per-markerset uniforms
        gl.uniform1i(gl.getUniformLocation(program, "u_transformIndex"),
            glUtils._collectionItemIndex[uid] != null ? glUtils._collectionItemIndex[uid] : -1);
        gl.uniform1f(gl.getUniformLocation(program, "u_markerScale"), glUtils._markerScaleFactor[uid]);
        gl.uniform1i(gl.getUniformLocation(program, "u_usePiechartFromMarker"), glUtils._usePiechartFromMarker[uid]);
        gl.uniform1i(gl.getUniformLocation(program, "u_useShapeFromMarker"), glUtils._useShapeFromMarker[uid]);
        gl.uniform1i(gl.getUniformLocation(program, "u_useAbsoluteMarkerSize"), glUtils._useAbsoluteMarkerSize[uid]);
        gl.uniform1f(gl.getUniformLocation(program, "u_markerOpacity"), glUtils._markerOpacity[uid]);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, glUtils._textures[uid + "_colorLUT"]);
        gl.uniform1i(gl.getUniformLocation(program, "u_colorLUT"), 0);

        // 1st pass: clear the corner pixel
        gl.uniform1i(gl.getUniformLocation(program, "u_op"), 0);
        gl.drawArrays(gl.POINTS, 0, 1);  // Note: this drawcall does not have to be indexed
        // 2nd pass: draw all the markers (as single pixels)
        gl.uniform1i(gl.getUniformLocation(program, "u_op"), 1);
        for (let offset = 0; offset < numPoints; offset += chunkSize) {
            const count = (offset + chunkSize >= numPoints) ? numPoints - offset : chunkSize;
            gl.drawArrays(gl.POINTS, offset, count);
        }

        // Read back pixel at location (0, 0) to get the picked object
        const result = new Uint8Array(4);
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, result);
        //const picked = Number(result[2] + result[1] * 256 + result[0] * 65536) - 1;
        const picked = Number(result[0] + result[1] * 256 + result[2] * 65536 + result[3] * 16777216) - 1;
        if (picked >= 0)
            glUtils._pickedMarker = [uid, picked];
    }

    // Restore render pipeline state
    gl.bindVertexArray(null);
    gl.useProgram(null);
}


/**
 * @summary Do rendering to the WebGL canvas.
 * Calling this function will force an update of the rendering of markers and
 * the data used for picking (i.e. for marker selection). Only marker datasets
 * for which glUtils.loadMarkers() have been called will be rendered.
 */
glUtils.draw = function() {
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl2", glUtils._options);
    const ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");

    // Update per-image transforms that take into account if collection mode
    // viewing is enabled for the image layers
    glUtils._updateTransformUBO(glUtils._buffers["transformUBO"]);

    // The OSD viewer can be rotated, so need to also apply rotation to markers
    const orientationDegrees = tmapp["ISS_viewer"].viewport.getRotation();
    const t = orientationDegrees * (3.141592 / 180.0);
    const viewportTransform = [Math.cos(t), -Math.sin(t), Math.sin(t), Math.cos(t)];

    // Compute adjusted marker scale so that the actual marker size becomes less
    // dependant on screen resolution or window size
    let markerScaleAdjusted = glUtils._markerScale;
    if (glUtils._useMarkerScaleFix) markerScaleAdjusted *= (gl.canvas.height / 900.0);
    markerScaleAdjusted /= tmapp["ISS_viewer"].viewport.getBounds().height;  // FIXME

    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    if (glUtils._pickingEnabled) {
        glUtils._drawPickingPass(gl, viewportTransform, markerScaleAdjusted);
        glUtils._pickingEnabled = false;  // Clear flag until next click event
    }

    const query = glUtils._query;
    if (glUtils._logPerformance) {
        if (query == null) {
            glUtils._query = gl.createQuery();
            gl.beginQuery(ext.TIME_ELAPSED_EXT, glUtils._query);
        }
    }

    glUtils._drawColorPass(gl, viewportTransform, markerScaleAdjusted);

    if (glUtils._logPerformance) {
        if (query == null) {
            gl.endQuery(ext.TIME_ELAPSED_EXT);
        } else {
            const available = gl.getQueryParameter(glUtils._query, gl.QUERY_RESULT_AVAILABLE);
            const disjoint = gl.getParameter(ext.GPU_DISJOINT_EXT);
            if (available && !disjoint) {
                const timeElapsed = gl.getQueryParameter(glUtils._query, gl.QUERY_RESULT);
                console.log("Rasterization time (GPU) (ms): ", timeElapsed / 1e6);
            }
            if (available || disjoint) {
                gl.deleteQuery(glUtils._query);
                glUtils._query = null;
            }
        }
    }
}


/**
 * @summary Do GPU-based picking for marker selection.
 * This function is a callback and should not normally be called directly. The
 * function will automatically call glUtils.draw() to update the rendering and
 * the picking.
 * @param {Object} event An object with click events from the canvas
 */
glUtils.pick = function(event) {
    if (event.quick) {
        glUtils._pickingEnabled = true;
        glUtils._pickingLocation = [event.position.x * glUtils._resolutionScaleActual,
                                    event.position.y * glUtils._resolutionScaleActual];
        glUtils.draw();  // This will update the value of glUtils._pickedMarker

        const pickedMarker = glUtils._pickedMarker;
        const hasPickedMarker = pickedMarker[1] >= 0;

        tmapp["ISS_viewer"].removeOverlay("ISS_marker_info");
        if (hasPickedMarker && glUtils._showMarkerInfo) {
            const uid = pickedMarker[0];
            const markerIndex = pickedMarker[1];
            const tabName = interfaceUtils.getElementById(uid + "_marker-tab-name").textContent;
            const markerData = dataUtils.data[uid]["_processeddata"];
            const keyName = dataUtils.data[uid]["_gb_col"];
            const groupName = (keyName != null) ? markerData[keyName][markerIndex] : undefined;
            const piechartPropertyName = dataUtils.data[uid]["_pie_col"];

            const div = document.createElement("div");
            div.id = "ISS_marker_info";
            div.width = "1px"; div.height = "1px";
            if (glUtils._usePiechartFromMarker[uid]) {
                div.innerHTML = markerUtils.makePiechartTable(uid, markerIndex, piechartPropertyName);
            } else {
                div.innerHTML = markerUtils.getMarkerTooltip(uid, markerIndex);
                console.log("Marker clicked:",tabName, groupName, "index:", markerIndex);
            }
            div.classList.add("viewer-layer", "m-0", "p-1");

            tmapp["ISS_viewer"].addOverlay({
                element: div,
                placement: "TOP_LEFT",
                location: tmapp["ISS_viewer"].viewport.viewerElementToViewportCoordinates(event.position),
                checkResize: false,
                rotationMode: OpenSeadragon.OverlayRotationMode.NO_ROTATION
            });
            interfaceUtils._mGenUIFuncs.ActivateTab(uid);
            var tr = document.querySelectorAll('[data-uid="'+uid+'"][data-key="'+groupName+'"]')[0];
            if (tr != null) {
                tr.scrollIntoView({block: "nearest",inline: "nearest"});
                tr.classList.remove("transition_background")
                tr.classList.add("table-primary")
                setTimeout(function(){tr.classList.add("transition_background");tr.classList.remove("table-primary");},400);
            }
        }

        const pickedRegion = regionUtils._findRegionByPoint(event.position);
        const hasPickedRegion = pickedRegion != null;
        glUtils._regionPicked = pickedRegion;

        tmapp["ISS_viewer"].removeOverlay("ISS_region_info");
        if (hasPickedRegion && regionUtils._regionMode == "select") {
            const div = document.createElement("div");
            div.id = "ISS_region_info";
            div.width = "1px"; div.height = "1px";
            div.innerHTML = regionUtils._regions[pickedRegion].regionName;
            console.log("Region clicked:", pickedRegion);
            div.classList.add("viewer-layer", "m-0", "p-1");
            div.style.zIndex = 99;

            tmapp["ISS_viewer"].addOverlay({
                element: div,
                placement: "TOP_RIGHT",
                location: tmapp["ISS_viewer"].viewport.viewerElementToViewportCoordinates(event.position),
                checkResize: false,
                rotationMode: OpenSeadragon.OverlayRotationMode.NO_ROTATION
            });
            if (regionUtils._selectedRegions[pickedRegion]) {
                regionUtils.deSelectRegion(pickedRegion);
            }
            else {
                if (!event.shift){
                    regionUtils.resetSelection();
                }
                regionUtils.selectRegion(regionUtils._regions[pickedRegion]);
            }
            console.log(event);
        }
        else {
            if (!event.shift){
                regionUtils.resetSelection();
            }
        }
    }
}


/**
 * @summary Callback for resizing the WebGL canvas.
 * Calling this function will force an update of the width and height of the
 * WebGL canvas, but will not automatically call glUtils.draw() to update the
 * rendering.
 */
glUtils.resize = function() {
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl2", glUtils._options);

    const op = tmapp["object_prefix"];
    const width = tmapp[op + "_viewer"].viewport.containerSize.x;
    const height = tmapp[op + "_viewer"].viewport.containerSize.y;

    glUtils._resolutionScaleActual = glUtils._resolutionScale * window.devicePixelRatio;
    if (Math.max(width, height) * glUtils._resolutionScale >= 4096.0) {
        // A too large WebGL canvas can lead to misalignment between the WebGL
        // markers and the OSD image layers, so here the resolution scaling
        // factor is adjusted to restrict the canvas size to a safe value
        glUtils._resolutionScaleActual *= 4096.0 / (Math.max(width, height) * glUtils._resolutionScale);
    }
    gl.canvas.width = width * glUtils._resolutionScaleActual;
    gl.canvas.height = height * glUtils._resolutionScaleActual;
}


/**
 * @summary Callback for resizing the WebGL canvas.
 * Works like glUtils.resize(), but will also automatically call glUtils.draw()
 * to update the rendering.
 */
glUtils.resizeAndDraw = function() {
    glUtils.resize();
    glUtils.draw();
}


/**
 * @summary Callback for updating marker scale when changing the global marker size GUI slider.
 * This function is a callback and should not normally be called directly.
 */
glUtils.updateMarkerScale = function() {
    const globalMarkerSize = Number(document.getElementById("ISS_globalmarkersize_text").value);
    // Clamp the scale factor to avoid giant markers and slow rendering if the
    // user inputs a very large value (say 10000 or something)
    glUtils._markerScale = Math.max(0.01, Math.min(20.0, globalMarkerSize / 25.0));
}


/**
 * @summary Callback for restoring WebGL resources after WebGL context is lost
 * This function is a callback and should not normally be called directly. Loss
 * of context can happen when for example the computer goes into sleep mode.
 */
glUtils.restoreLostContext = function(event) {
    console.log("Restoring WebGL objects after context loss");
    let canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl2", glUtils._options);

    // Restore shared WebGL objects
    glUtils._programs["markers"] = glUtils._loadShaderProgram(gl, glUtils._markersVS, glUtils._markersFS);
    glUtils._programs["markers_instanced"] = glUtils._loadShaderProgram(gl, glUtils._markersVS, glUtils._markersFS, "#define USE_INSTANCING\n");
    glUtils._programs["picking"] = glUtils._loadShaderProgram(gl, glUtils._pickingVS, glUtils._pickingFS);
    glUtils._programs["edges"] = glUtils._loadShaderProgram(gl, glUtils._edgesVS, glUtils._edgesFS);
    glUtils._programs["regions"] = glUtils._loadShaderProgram(gl, glUtils._regionsVS, glUtils._regionsFS);
    glUtils._textures["shapeAtlas"] = glUtils._loadTextureFromImageURL(gl, glUtils._markershapes);
    glUtils._buffers["quad"] = glUtils._createQuad(gl);
    glUtils._buffers["transformUBO"] = glUtils._createUniformBuffer(gl);
    glUtils._textures["regionLUT"] = glUtils._createRegionLUTTexture(gl, glUtils._regionMaxNumRegions);
    glUtils._vaos["empty"] = gl.createVertexArray();

    // Restore per-markerset WebGL objects
    for (let [uid, numPoints] of Object.entries(glUtils._numPoints)) {
        delete glUtils._buffers[uid + "_markers"];
        delete glUtils._buffers[uid + "_markers_secondary"];
        delete glUtils._vaos[uid + "_markers"];
        delete glUtils._vaos[uid + "_markers_instanced"];
        delete glUtils._buffers[uid + "_edges"];
        delete glUtils._vaos[uid + "_edges"];
        delete glUtils._textures[uid + "_colorLUT"];
        delete glUtils._textures[uid + "_colorscale"];
        glUtils.loadMarkers(uid);
    }

    // Restore per-layer WebGL objects for drawing regions
    for (let key in Object.keys(glUtils._textures)) {
        // Check all named texture objects to be safe, since the regions
        // a texture was created for might have been deleted or moved to a
        // different image layer (been assigned a different collectionIndex)
        if (key.includes("regionData")) delete glUtils._textures[key];
    }
    if (Object.keys(regionUtils._edgeListsByLayer).length) {
        glUtils.updateRegionDataTextures();
        glUtils.updateRegionLUTTextures();
    }

    glUtils.draw();  // Make sure markers and other objects are redrawn
}


/**
 * @summary Do initialization of the WebGL canvas.
 * This will also load WebGL resources like shaders and textures, as well as set
 * up events for interaction with other parts of TissUUmaps such as the
 * OpenSeaDragon (OSD) canvas.
 */
glUtils.init = function() {
    if (glUtils._initialized) return;

    let canvas = document.getElementById("gl_canvas");
    if (!canvas) canvas = this._createMarkerWebGLCanvas();
    canvas.addEventListener("webglcontextlost", function(e) { e.preventDefault(); }, false);
    canvas.addEventListener("webglcontextrestored", glUtils.restoreLostContext, false);
    const gl = canvas.getContext("webgl2", glUtils._options);

    if (!(gl instanceof WebGL2RenderingContext)) {
        interfaceUtils.alert("Error: TissUUmaps requires a web browser that supports WebGL 2.0");
    }

    // Get HW capabilities from WebGL context
    glUtils._caps[gl.MAX_TEXTURE_SIZE] = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    glUtils._caps[gl.ALIASED_POINT_SIZE_RANGE] = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
    console.assert(glUtils._caps[gl.ALIASED_POINT_SIZE_RANGE] instanceof Float32Array);

    // Disable instanced marker drawing by default if the HW point size limit
    // is large enough. Should be faster in most cases, and we can still
    // temporarily switch to instanced drawing during viewport captures to
    // avoid the HW point size limit.
    if (glUtils._caps[gl.ALIASED_POINT_SIZE_RANGE][1] >= 1023) {
        glUtils._useInstancing = false;
    }

    // Place marker canvas under the OSD canvas. Doing this also enables proper
    // compositing with the minimap and other OSD elements.
    const osd = document.getElementsByClassName("openseadragon-canvas")[0];
    osd.appendChild(canvas);

    this._programs["markers"] = this._loadShaderProgram(gl, this._markersVS, this._markersFS);
    this._programs["markers_instanced"] = this._loadShaderProgram(gl, this._markersVS, this._markersFS, "#define USE_INSTANCING\n");
    this._programs["picking"] = this._loadShaderProgram(gl, this._pickingVS, this._pickingFS);
    this._programs["edges"] = this._loadShaderProgram(gl, this._edgesVS, this._edgesFS);
    this._programs["regions"] = this._loadShaderProgram(gl, this._regionsVS, this._regionsFS);
    this._textures["shapeAtlas"] = this._loadTextureFromImageURL(gl, glUtils._markershapes);
    this._buffers["quad"] = this._createQuad(gl);
    this._buffers["transformUBO"] = this._createUniformBuffer(gl);
    this._textures["regionLUT"] = this._createRegionLUTTexture(gl, glUtils._regionMaxNumRegions);
    this._vaos["empty"] = gl.createVertexArray();

    this._createColorbarCanvas();  // The colorbar is drawn separately in a 2D-canvas

    glUtils.updateMarkerScale();
    document.getElementById("ISS_globalmarkersize_text").addEventListener("input", glUtils.updateMarkerScale);
    document.getElementById("ISS_globalmarkersize_text").addEventListener("input", glUtils.draw);

    tmapp["hideSVGMarkers"] = true;
    tmapp["ISS_viewer"].removeHandler('resize', glUtils.resizeAndDraw);
    tmapp["ISS_viewer"].addHandler('resize', glUtils.resizeAndDraw);
    tmapp["ISS_viewer"].removeHandler('open', glUtils.draw);
    tmapp["ISS_viewer"].addHandler('open', glUtils.draw);
    tmapp["ISS_viewer"].removeHandler('viewport-change', glUtils.draw);
    tmapp["ISS_viewer"].addHandler('viewport-change', glUtils.draw);
    tmapp["ISS_viewer"].removeHandler('canvas-click', glUtils.pick);
    tmapp["ISS_viewer"].addHandler('canvas-click', glUtils.pick);

    glUtils._initialized = true;
    glUtils.resize();  // Force initial resize to OSD canvas size
}
