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

    // Marker settings and info stored per UID (this could perhaps be
    // better handled by having an object per UID that stores all info
    // and is easy to delete when closing a marker tab...)
    _numPoints: {},              // {uid: numPoints, ...}
    _numEdges: {},               // {uid: numEdges, ...}
    _zOrder: {},                 // {uid: float, ...}
    _markerScalarRange: {},      // {uid: [minval, maxval], ...}
    _markerScaleFactor: {},      // {uid: float}
    _markerScalarPropertyName: {},  // {uid: string, ...}
    _markerOpacity: {},          // {uid: alpha, ...}
    _markerOutline: {},          // {uid: boolean, ...}
    _useColorFromMarker: {},     // {uid: boolean, ...}
    _useColorFromColormap: {},   // {uid: boolean, ...}
    _useScaleFromMarker: {},     // {uid: boolean, ...}
    _useOpacityFromMarker: {},   // {uid: boolean, ...}
    _usePiechartFromMarker: {},  // {uid: boolean, ...}
    _useShapeFromMarker: {},     // {uid: boolean, ...}
    _useSortByCol: {},           // {uid: boolean, ...}
    _colorscaleName: {},         // {uid: colorscaleName, ...}
    _colorscaleData: {},         // {uid: array of RGBA values, ...}
    _barcodeToLUTIndex: {},      // {uid: dict, ...}
    _barcodeToKey: {},           // {uid: dict, ...}
    _collectionItemIndex: {},    // {uid: number, ...}
    _markerInputsCached: {},     // {uid: dict, ...}
    _edgeInputsCached: {},       // {uid: dict, ...}

    // Global marker settings and info
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
    _logPerformance: false,       // Use GPU timer queries to log performance
    _piechartPalette: ["#fff100", "#ff8c00", "#e81123", "#ec008c", "#68217a", "#00188f", "#00bcf2", "#00b294", "#009e49", "#bad80a"]
}


glUtils._markersVS = `
    #define SHAPE_INDEX_CIRCLE 7.0
    #define SHAPE_INDEX_CIRCLE_NOSTROKE 16.0
    #define SHAPE_GRID_SIZE 4.0
    #define MAX_NUM_IMAGES 192
    #define DISCARD_VERTEX { gl_Position = vec4(2.0, 2.0, 2.0, 0.0); return; }

    uniform mat2 u_viewportTransform;
    uniform vec2 u_canvasSize;
    uniform float u_transformIndex;
    uniform float u_markerScale;
    uniform float u_globalMarkerScale;
    uniform vec2 u_markerScalarRange;
    uniform float u_markerOpacity;
    uniform float u_maxPointSize;
    uniform bool u_useColorFromMarker;
    uniform bool u_useColorFromColormap;
    uniform bool u_usePiechartFromMarker;
    uniform bool u_useShapeFromMarker;
    uniform bool u_alphaPass;
    uniform int u_pickedMarker;
    uniform sampler2D u_colorLUT;
    uniform sampler2D u_colorscale;

    layout(std140) uniform TransformUniforms {
        mat3x2 imageToViewport[MAX_NUM_IMAGES];
    } u_transformUBO;

    layout(location = 0) in vec4 in_position;
    layout(location = 1) in int in_index;
    layout(location = 2) in float in_scale;
    layout(location = 3) in float in_shape;
    layout(location = 4) in float in_opacity;
    layout(location = 5) in float in_transform;

    out vec4 v_color;
    out vec2 v_shapeOrigin;
    out vec2 v_shapeSector;
    out float v_shapeSize;
    #ifdef USE_INSTANCING
    out highp vec2 v_texCoord;
    #endif  // USE_INSTANCING

    vec3 hex_to_rgb(float v)
    {
        // Extract RGB color from 24-bit hex color stored in float
        v = clamp(v, 0.0, 16777215.0);
        return floor(mod((v + 0.49) / vec3(65536.0, 256.0, 1.0), 256.0)) / 255.0;
    }

    void main()
    {
        float transformIndex = u_transformIndex >= 0.0 ? u_transformIndex : in_transform;
        mat3x2 imageToViewport = u_transformUBO.imageToViewport[int(transformIndex)];
        vec2 viewportPos = imageToViewport * vec3(in_position.xy, 1.0);
        vec2 ndcPos = viewportPos * 2.0 - 1.0;
        ndcPos.y = -ndcPos.y;
        ndcPos = u_viewportTransform * ndcPos;

        float lutIndex = mod(in_position.z, 4096.0);
        v_color = texture(u_colorLUT, vec2(lutIndex / 4095.0, 0.5));

        if (u_useColorFromMarker || u_useColorFromColormap) {
            vec2 range = u_markerScalarRange;
            float normalized = (in_position.w - range[0]) / (range[1] - range[0]);
            v_color.rgb = texture(u_colorscale, vec2(normalized, 0.5)).rgb;
            if (u_useColorFromMarker) v_color.rgb = hex_to_rgb(in_position.w);
        }

        if (u_useShapeFromMarker && v_color.a > 0.0) {
            // Add one to marker index and normalize, to make things consistent
            // with how marker visibility and shape is stored in the LUT
            v_color.a = (floor(in_position.z / 4096.0) + 1.0) / 255.0;
        }

        if (u_usePiechartFromMarker && v_color.a > 0.0) {
            v_shapeSector[0] = mod(in_shape, 4096.0) / 4095.0;
            v_shapeSector[1] = floor(in_shape / 4096.0) / 4095.0;
            v_color.rgb = hex_to_rgb(in_position.w);
            v_color.a = SHAPE_INDEX_CIRCLE_NOSTROKE / 255.0;
            if (u_pickedMarker == in_index) v_color.a = SHAPE_INDEX_CIRCLE / 255.0;

            // For the alpha pass, we only want to draw the marker once
            float sectorIndex = floor(in_position.z / 4096.0);
            if (u_alphaPass) v_color.a *= float(sectorIndex == 0.0);
        }

        gl_Position = vec4(ndcPos, 0.0, 1.0);
        gl_PointSize = in_scale * u_markerScale * u_globalMarkerScale;
        float alphaFactorSize = clamp(gl_PointSize, 0.2, 1.0); 
        gl_PointSize = clamp(gl_PointSize, 1.0, u_maxPointSize);

        v_shapeOrigin.x = mod((v_color.a + 0.00001) * 255.0 - 1.0, SHAPE_GRID_SIZE);
        v_shapeOrigin.y = floor(((v_color.a + 0.00001) * 255.0 - 1.0) / SHAPE_GRID_SIZE);
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
        if (v_color.a == 0.0) DISCARD_VERTEX;
    }
`;


glUtils._markersFS = `
    #define UV_SCALE 0.7
    #define SHAPE_GRID_SIZE 4.0

    precision mediump float;

    uniform bool u_markerOutline;
    uniform bool u_usePiechartFromMarker;
    uniform bool u_alphaPass;
    uniform sampler2D u_shapeAtlas;

    in vec4 v_color;
    in vec2 v_shapeOrigin;
    in vec2 v_shapeSector;
    in float v_shapeSize;
    #ifdef USE_INSTANCING
    in highp vec2 v_texCoord;
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
    #ifdef USE_INSTANCING
        vec2 uv = (v_texCoord - 0.5) * UV_SCALE + 0.5;
    #else
        vec2 uv = (gl_PointCoord - 0.5) * UV_SCALE + 0.5;
    #endif  // USE_INSTANCING
        uv = (uv + v_shapeOrigin) * (1.0 / SHAPE_GRID_SIZE);

        // Sample shape texture in which the blue channel encodes alpha and the
        // red and green channels encode grayscale for marker shape with and
        // without outline, respectively
        vec4 shapeColor = texture(u_shapeAtlas, uv, -0.5);
        shapeColor = u_markerOutline ? shapeColor.rrrb : shapeColor.gggb;

        if (u_usePiechartFromMarker && !u_alphaPass) {
            float delta = 0.25 / v_shapeSize;
        #ifdef USE_INSTANCING
            shapeColor.a *= sectorToAlphaAA(v_shapeSector, v_texCoord, delta);
        #else
            shapeColor.a *= sectorToAlphaAA(v_shapeSector, gl_PointCoord, delta);
        #endif  // USE_INSTANCING
        }

        out_color = shapeColor * v_color;
        if (out_color.a < 0.004) discard;
    }
`;


glUtils._pickingVS = `
    #define UV_SCALE 0.7
    #define SHAPE_INDEX_CIRCLE_NOSTROKE 16.0
    #define SHAPE_GRID_SIZE 4.0
    #define MAX_NUM_IMAGES 192
    #define DISCARD_VERTEX { gl_Position = vec4(2.0, 2.0, 2.0, 0.0); return; }

    #define OP_CLEAR 0
    #define OP_WRITE_INDEX 1

    uniform mat2 u_viewportTransform;
    uniform vec2 u_canvasSize;
    uniform vec2 u_pickingLocation;
    uniform float u_transformIndex;
    uniform float u_markerScale;
    uniform float u_globalMarkerScale;
    uniform float u_markerOpacity;
    uniform float u_maxPointSize;
    uniform bool u_usePiechartFromMarker;
    uniform bool u_useShapeFromMarker;
    uniform int u_op;
    uniform sampler2D u_colorLUT;
    uniform sampler2D u_shapeAtlas;

    layout(std140) uniform TransformUniforms {
        mat3x2 imageToViewport[MAX_NUM_IMAGES];
    } u_transformUBO;

    layout(location = 0) in vec4 in_position;
    layout(location = 1) in int in_index;
    layout(location = 2) in float in_scale;
    layout(location = 4) in float in_opacity;
    layout(location = 5) in float in_transform;

    out vec4 v_color;

    vec3 hex_to_rgb(float v)
    {
        // Extract RGB color from 24-bit hex color stored in float
        v = clamp(v, 0.0, 16777215.0);
        return floor(mod((v + 0.49) / vec3(65536.0, 256.0, 1.0), 256.0)) / 255.0;
    }

    void main()
    {
        float transformIndex = u_transformIndex >= 0.0 ? u_transformIndex : in_transform;
        mat3x2 imageToViewport = u_transformUBO.imageToViewport[int(transformIndex)];
        vec2 viewportPos = imageToViewport * vec3(in_position.xy, 1.0);
        vec2 ndcPos = viewportPos * 2.0 - 1.0;
        ndcPos.y = -ndcPos.y;
        ndcPos = u_viewportTransform * ndcPos;

        v_color = vec4(0.0);
        if (u_op == OP_WRITE_INDEX) {
            float lutIndex = mod(in_position.z, 4096.0);
            float shapeID = texture(u_colorLUT, vec2(lutIndex / 4095.0, 0.5)).a;
            if (shapeID == 0.0) DISCARD_VERTEX;

            if (u_useShapeFromMarker) {
                // Add one to marker index and normalize, to make things consistent
                // with how marker visibility and shape is stored in the LUT
                shapeID = (floor(in_position.z / 4096.0) + 1.0) / 255.0;
            }

            if (u_usePiechartFromMarker) {
                shapeID = SHAPE_INDEX_CIRCLE_NOSTROKE / 255.0;

                // For the picking pass, we only want to draw the marker once
                float sectorIndex = floor(in_position.z / 4096.0);
                if (sectorIndex > 0.0) DISCARD_VERTEX;
            }

            vec2 canvasPos = (ndcPos * 0.5 + 0.5) * u_canvasSize;
            canvasPos.y = (u_canvasSize.y - canvasPos.y);  // Y-axis is inverted
            float pointSize = in_scale * u_markerScale * u_globalMarkerScale;
            pointSize = clamp(pointSize, 2.0, u_maxPointSize);

            // Do coarse inside/outside test against bounding box for marker
            vec2 uv = (canvasPos - u_pickingLocation) / pointSize + 0.5;
            uv.y = (1.0 - uv.y);  // Flip y-axis to match gl_PointCoord behaviour
            if (abs(uv.x - 0.5) > 0.5 || abs(uv.y - 0.5) > 0.5) DISCARD_VERTEX;

            // Do fine-grained inside/outside test by sampling the shape texture
            // with alpha encoded in the blue channel
            vec2 shapeOrigin = vec2(0.0);
            shapeOrigin.x = mod((shapeID + 0.00001) * 255.0 - 1.0, SHAPE_GRID_SIZE);
            shapeOrigin.y = floor(((shapeID + 0.00001) * 255.0 - 1.0) / SHAPE_GRID_SIZE);
            uv = (uv - 0.5) * UV_SCALE + 0.5;
            uv = (uv + shapeOrigin) * (1.0 / SHAPE_GRID_SIZE);
            if (texture(u_shapeAtlas, uv).b < 0.5) DISCARD_VERTEX;

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
    precision mediump float;

    in vec4 v_color;

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
    uniform float u_transformIndex;
    uniform float u_markerScale;
    uniform float u_globalMarkerScale;
    uniform float u_markerOpacity;
    uniform float u_maxPointSize;
    uniform float u_edgeThicknessRatio;
    uniform sampler2D u_colorLUT;

    layout(std140) uniform TransformUniforms {
        mat3x2 imageToViewport[MAX_NUM_IMAGES];
    } u_transformUBO;

    layout(location = 0) in vec4 in_position;
    layout(location = 1) in int in_index;
    layout(location = 4) in float in_opacity;
    layout(location = 5) in float in_transform;

    out vec4 v_color;
    out highp vec2 v_texCoord;

    void main()
    {
        float transformIndex0 = u_transformIndex >= 0.0 ? u_transformIndex : mod(in_transform, 256.0);
        float transformIndex1 = u_transformIndex >= 0.0 ? u_transformIndex : floor(in_transform / 256.0);
        mat3x2 imageToViewport0 = u_transformUBO.imageToViewport[int(transformIndex0)];
        mat3x2 imageToViewport1 = u_transformUBO.imageToViewport[int(transformIndex1)];

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
    precision mediump float;

    in vec4 v_color;
    in highp vec2 v_texCoord;

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


glUtils._createMarkerBuffer = function(gl, numBytes) {
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
    
    const sectorsPropertyName = newInputs.sectorsPropertyName = dataUtils.data[uid]["_pie_col"];
    const usePiechartFromMarker = dataUtils.data[uid]["_pie_col"] != null;
    if (dataUtils.data[uid]["_pie_dict"] && sectorsPropertyName) {
        glUtils._piechartPalette = JSON.parse(dataUtils.data[uid]["_pie_dict"])
        if (typeof glUtils._piechartPalette === "object") {
            glUtils._piechartPalette = sectorsPropertyName.split(";").map(function(sector) {
                return glUtils._piechartPalette[sector];
            })
        }
    }
    const piechartPalette = glUtils._piechartPalette;
    let numSectors = 1;

    const shapePropertyName = newInputs.shapePropertyName = dataUtils.data[uid]["_shape_col"];
    const useShapeFromMarker = newInputs.useShapeFromMarker = dataUtils.data[uid]["_shape_col"] != null;
    const numShapes = Object.keys(markerUtils._symbolStrings).length;
    let shapeIndex = 0;

    const opacityPropertyName = newInputs.opacityPropertyName = dataUtils.data[uid]["_opacity_col"];
    const useOpacityFromMarker = newInputs.useOpacityFromMarker = dataUtils.data[uid]["_opacity_col"] != null;
    const markerOpacityFactor = dataUtils.data[uid]["_opacity"];

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
    // changes to the format! Two buffers will be used for the vertex data
    // to avoid the 1GB max size limit imposed by QtWebEngine/Chromium.
    const NUM_BYTES_PER_MARKER = 16;
    const NUM_BYTES_PER_MARKER_SECONDARY = 16;
    const POINT_OFFSET = numPoints * 0,
          INDEX_OFFSET = numPoints * 0,
          SCALE_OFFSET = numPoints * 4,
          SHAPE_OFFSET = numPoints * 8;
          OPACITY_OFFSET = numPoints * 12;
          TRANSFORM_OFFSET = numPoints * 14;
    const POINT_LOCATION = 0,
          INDEX_LOCATION = 1,
          SCALE_LOCATION = 2,
          SHAPE_LOCATION = 3,
          OPACITY_LOCATION = 4;
          TRANSFORM_LOCATION = 5;

    const lastInputs = glUtils._markerInputsCached[uid];
    if (forceUpdate || (lastInputs != JSON.stringify(newInputs)) || dataUtils.data[uid]["modified"]) {
        dataUtils.data[uid]["modified"] = false;
        forceUpdate = true;
        scalarRange = [1e9, -1e9];  // This range will be computed from the data

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
                numSectors = markerData[sectorsPropertyName][0].split(";").length;
                bytedata_point = new Float32Array(chunkSize * numSectors * 4);
                bytedata_index = new Int32Array(chunkSize * numSectors * 1);
                bytedata_scale = new Float32Array(chunkSize * numSectors * 1);
                bytedata_shape = new Float32Array(chunkSize * numSectors * 1);
                bytedata_opacity = new Uint16Array(chunkSize * numSectors * 1);
                bytedata_transform = new Uint16Array(chunkSize * numSectors * 1);

                for (let i = 0; i < chunkSize; ++i) {
                    const markerIndex = i + offset;
                    const sectors = markerData[sectorsPropertyName][markerIndex].split(";");
                    const piechartAngles = glUtils._createPiechartAngles(sectors);
                    const lutIndex = (keyName != null) ? barcodeToLUTIndex[markerData[keyName][markerIndex]] : 0;
                    const opacity = useOpacityFromMarker ? markerData[opacityPropertyName][markerIndex] : 1.0;
                    if (useCollectionItemFromMarker) collectionItemIndex = markerData[collectionItemPropertyName][markerIndex];

                    for (let j = 0; j < numSectors; ++j) {
                        const k = (i * numSectors + j);
                        const sectorIndex = j;
                        hexColor = piechartPalette[j % piechartPalette.length];

                        bytedata_point[4 * k + 0] = markerData[xPosName][markerIndex];
                        bytedata_point[4 * k + 1] = markerData[yPosName][markerIndex];
                        bytedata_point[4 * k + 2] = lutIndex + sectorIndex * 4096.0;
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
                    const markerIndex = i + offset;
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
                    bytedata_point[4 * i + 2] = lutIndex + Number(shapeIndex) * 4096.0;
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
                glUtils._buffers[uid + "_markers"] = glUtils._createMarkerBuffer(
                    gl, numPoints * numSectors * NUM_BYTES_PER_MARKER);
            if (!(uid + "_markers_secondary" in glUtils._buffers))
                glUtils._buffers[uid + "_markers_secondary"] = glUtils._createMarkerBuffer(
                    gl, numPoints * numSectors * NUM_BYTES_PER_MARKER_SECONDARY);
            if (!(uid + "_markers_indices" in glUtils._buffers))
                glUtils._buffers[uid + "_markers_indices"] = glUtils._createIndexBuffer(
                    gl, numPoints * numSectors * 4);  // Use 32-bits indices
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
                {
                    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, glUtils._buffers[uid + "_markers_indices"]);
                    const newBufferSize = numPoints * numSectors * 4;  // Use 32-bits indices
                    const oldBufferSize = gl.getBufferParameter(gl.ELEMENT_ARRAY_BUFFER, gl.BUFFER_SIZE);
                    if (newBufferSize != oldBufferSize)
                        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, newBufferSize, gl.STATIC_DRAW);
                    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
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

        console.time("Generate index data");
        {
            const numIndices = numPoints * numSectors;
            const bytedata_indices = new Uint32Array(numIndices);
            for (let index = 0; index < numIndices; ++index) {
                bytedata_indices[index] = index;
            }
            
            if (useSortByCol) {
                const colData = markerData[sortByCol];
                if (sortByDesc) {
                    // Sort in descending order
                    bytedata_indices.sort((i, j) => Number(colData[j]) - Number(colData[i]));
                } else {  // Sort in ascending order
                    bytedata_indices.sort((i, j) => Number(colData[i]) - Number(colData[j]));
                }
            }

            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, glUtils._buffers[uid + "_markers_indices"]);
            gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, bytedata_indices);
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
        }
        console.timeEnd("Generate index data");

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
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, glUtils._buffers[uid + "_markers_indices"]);
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
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, glUtils._buffers[uid + "_markers_indices"]);
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
    glUtils._markerOutline[uid] = markerOutline;
    glUtils._useColorFromMarker[uid] = useColorFromMarker;
    glUtils._useColorFromColormap[uid] = useColorFromColormap;
    glUtils._useScaleFromMarker[uid] = useScaleFromMarker;
    glUtils._useOpacityFromMarker[uid] = useOpacityFromMarker;
    glUtils._usePiechartFromMarker[uid] = usePiechartFromMarker;
    glUtils._useShapeFromMarker[uid] = useShapeFromMarker;
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
    delete glUtils._markerOutline[uid];
    delete glUtils._useColorFromMarker[uid];
    delete glUtils._useColorFromColormap[uid];
    delete glUtils._useScaleFromMarker[uid];
    delete glUtils._useOpacityFromMarker[uid];
    delete glUtils._usePiechartFromMarker[uid];
    delete glUtils._useShapeFromMarker[uid];
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
    gl.deleteBuffer(glUtils._buffers[uid + "_markers_indices"]);
    gl.deleteVertexArray(glUtils._vaos[uid + "_markers"]);
    gl.deleteVertexArray(glUtils._vaos[uid + "_markers_instanced"]);
    gl.deleteBuffer(glUtils._buffers[uid + "_edges"]);
    gl.deleteVertexArray(glUtils._vaos[uid + "_edges"]);
    gl.deleteTexture(glUtils._textures[uid + "_colorLUT"]);
    gl.deleteTexture(glUtils._textures[uid + "_colorscale"]);
    delete glUtils._buffers[uid + "_markers"];
    delete glUtils._buffers[uid + "_markers_secondary"];
    delete glUtils._buffers[uid + "_markers_indices"];
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
          OPACITY_LOCATION = 4;
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
                    bytedata_index[k] = lutIndex + (lutIndex_j * 4096.0);
                    bytedata_opacity[k] = Math.floor(Math.max(0.0, Math.min(1.0, opacity)) * 65535.0);
                    bytedata_transform[k] = collectionItemIndex + (collectionItemIndex_j * 256);
                }

                offsetEdges2 += edges.length;
            }

            // Create WebGL objects (if this has not already been done)
            if (!(uid + "_edges" in glUtils._buffers))
                glUtils._buffers[uid + "_edges"] = glUtils._createMarkerBuffer(gl, numEdges * NUM_BYTES_PER_EDGE);
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


// TODO Fix naming of this function, since we now use it for generic markers
glUtils._updateBarcodeToLUTIndexDict = function (uid, markerData, keyName) {
    const barcodeToLUTIndex = {};
    const barcodeToKey = {};
    const numPoints = markerData[markerData.columns[0]].length;
    console.log("Key name: " + keyName);
    for (let i = 0, index = 0; i < numPoints; ++i) {
        const barcode = (keyName != null) ? markerData[keyName][i] : undefined;
        if (!(barcode in barcodeToLUTIndex)) {
            barcodeToLUTIndex[barcode] = index++;
            barcodeToKey[barcode] = barcode;
            index = index % 4096;  // Prevent index from becoming >= the maximum LUT size,
                                   // since this causes problems with pie-chart markers
        }
    }
    glUtils._barcodeToLUTIndex[uid] = barcodeToLUTIndex;
    glUtils._barcodeToKey[uid] = barcodeToKey;
    console.log("barcodeToLUTIndex, barcodeToKey", barcodeToLUTIndex, barcodeToKey);
}


glUtils._createColorLUTTexture = function(gl) {
    const randomColors = [];
    for (let i = 0; i < 4096; ++i) {
        randomColors[4 * i + 0] = Math.random() * 256.0; 
        randomColors[4 * i + 1] = Math.random() * 256.0;
        randomColors[4 * i + 2] = Math.random() * 256.0;
        randomColors[4 * i + 3] = Math.floor(Math.random() * 7) + 1;
    }

    const bytedata = new Uint8Array(randomColors);

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); 
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 4096, 1, 0, gl.RGBA,
                  gl.UNSIGNED_BYTE, bytedata);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return texture;
}


glUtils._updateColorLUTTexture = function(gl, uid, texture) {
    if (!(uid + "_colorLUT" in glUtils._textures)) return;

    const hasGroups = dataUtils.data[uid]["_gb_col"] != null;

    const colors = new Array(4096 * 4);
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
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 4096, 1, 0, gl.RGBA,
                  gl.UNSIGNED_BYTE, bytedata);
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
    const imageTransforms = new Array(256 * 12).fill(0);
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

        // Construct 3x2 matrix (in col-major order) to be applied to marker positions.
        // Note: each col in the matrix must be padded to a vec4, because of std140
        // alignment rules for storing 3x2 matrices in arrays in UBOs.
        imageTransforms[i * 12 + 0] = flip * Math.cos(theta) * scaleX;
        imageTransforms[i * 12 + 4] = -Math.sin(theta) * scaleX;
        imageTransforms[i * 12 + 8] = shiftX - k0 * (scaleX * imageWidth) * Math.cos(theta) + k1 * (scaleX * imageHeight) * Math.sin(theta);
        imageTransforms[i * 12 + 1] = flip * Math.sin(theta) * scaleY;
        imageTransforms[i * 12 + 5] = Math.cos(theta) * scaleY;
        imageTransforms[i * 12 + 9] = shiftY - k0 * (scaleY * imageWidth) * Math.sin(theta) - k1 * (scaleY * imageHeight) * Math.cos(theta);
    }

    const bytedata = new Float32Array(imageTransforms);

    gl.bindBuffer(gl.UNIFORM_BUFFER, buffer);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, bytedata);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
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

    // Draw markers and edges interleaved, to ensure correct overlap per UID
    for (let uid of drawOrder) {
        if (glUtils._showEdgesExperimental) {
            glUtils._drawEdgesByUID(gl, viewportTransform, markerScaleAdjusted, uid);
        }
        glUtils._drawMarkersByUID(gl, viewportTransform, markerScaleAdjusted, uid);
    }
}


glUtils._drawMarkersByUID = function(gl, viewportTransform, markerScaleAdjusted, uid) {
    const numPoints = glUtils._numPoints[uid];
    if (numPoints == 0) return;

    // Note: marker rendering is currently broken when both instancing and
    // sorting are enabled at the same time. In that case, as workaround,
    // we will fall back to using point sprites instead.
    const useInstancing = glUtils._useInstancing && !glUtils._useSortByCol[uid];

    // Set up render pipeline
    const program = glUtils._programs[useInstancing ? "markers_instanced" : "markers"];
    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    // Set per-scene uniforms
    gl.uniformMatrix2fv(gl.getUniformLocation(program, "u_viewportTransform"), false, viewportTransform);
    gl.uniform2fv(gl.getUniformLocation(program, "u_canvasSize"), [gl.canvas.width, gl.canvas.height]);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale"), markerScaleAdjusted);
    gl.uniform1f(gl.getUniformLocation(program, "u_maxPointSize"), useInstancing ? 2048 : 256);
    gl.uniformBlockBinding(program, gl.getUniformBlockIndex(program, "TransformUniforms"), 0);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, glUtils._buffers["transformUBO"]);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, glUtils._textures["shapeAtlas"]);
    gl.uniform1i(gl.getUniformLocation(program, "u_shapeAtlas"), 2);

    gl.bindVertexArray(glUtils._vaos[uid + (useInstancing ? "_markers_instanced" : "_markers")]);

    // Set per-markerset uniforms
    gl.uniform1f(gl.getUniformLocation(program, "u_transformIndex"),
        glUtils._collectionItemIndex[uid] != null ? glUtils._collectionItemIndex[uid] : -1);
    gl.uniform1f(gl.getUniformLocation(program, "u_globalMarkerScale"), glUtils._globalMarkerScale * glUtils._markerScaleFactor[uid]);
    gl.uniform2fv(gl.getUniformLocation(program, "u_markerScalarRange"), glUtils._markerScalarRange[uid]);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerOpacity"), glUtils._markerOpacity[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_markerOutline"), glUtils._markerOutline[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_useColorFromMarker"), glUtils._useColorFromMarker[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_useColorFromColormap"), glUtils._useColorFromColormap[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_usePiechartFromMarker"), glUtils._usePiechartFromMarker[uid]);
    gl.uniform1i(gl.getUniformLocation(program, "u_useShapeFromMarker"), glUtils._useShapeFromMarker[uid]);
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
        if (useInstancing) {
            gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, numPoints);
        } else {
            gl.drawElements(gl.POINTS, numPoints, gl.UNSIGNED_INT, 0);
        }
        // 2nd pass: draw colors for individual piechart sectors
        gl.uniform1i(gl.getUniformLocation(program, "u_alphaPass"), false);
        gl.colorMask(true, true, true, false);
        if (useInstancing) {
            gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, numPoints);
        } else {
            gl.drawElements(gl.POINTS, numPoints, gl.UNSIGNED_INT, 0);
        }
        gl.colorMask(true, true, true, true);
    } else {
        if (useInstancing) {
            // Note: drawElementsInstanced is for some reason much slower than
            // drawArraysInstanced. So since sorting currently does not work
            // with instancing, we can use faster non-indexed drawing here.
            gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, numPoints);
        } else {
            gl.drawElements(gl.POINTS, numPoints, gl.UNSIGNED_INT, 0);
        }
    }

    // Restore render pipeline state
    gl.bindVertexArray(null);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.disable(gl.BLEND);
    gl.useProgram(null);
}


glUtils._drawEdgesByUID = function(gl, viewportTransform, markerScaleAdjusted, uid) {
    const numEdges = glUtils._numEdges[uid];
    if (numEdges == 0) return;

    // Set up render pipeline
    const program = glUtils._programs["edges"];
    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    // Set per-scene uniforms
    gl.uniformMatrix2fv(gl.getUniformLocation(program, "u_viewportTransform"), false, viewportTransform);
    gl.uniform2fv(gl.getUniformLocation(program, "u_canvasSize"), [gl.canvas.width, gl.canvas.height]);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale"), markerScaleAdjusted);
    gl.uniform1f(gl.getUniformLocation(program, "u_maxPointSize"), glUtils._useInstancing ? 2048 : 256);
    gl.uniform1f(gl.getUniformLocation(program, "u_edgeThicknessRatio"), glUtils._edgeThicknessRatio);
    gl.uniformBlockBinding(program, gl.getUniformBlockIndex(program, "TransformUniforms"), 0);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, glUtils._buffers["transformUBO"]);

    gl.bindVertexArray(glUtils._vaos[uid + "_edges"]);

    // Set per-markerset uniforms
    gl.uniform1f(gl.getUniformLocation(program, "u_globalMarkerScale"), glUtils._globalMarkerScale * glUtils._markerScaleFactor[uid]);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerOpacity"), glUtils._markerOpacity[uid]);
    gl.uniform1f(gl.getUniformLocation(program, "u_transformIndex"),
        glUtils._collectionItemIndex[uid] != null ? glUtils._collectionItemIndex[uid] : -1);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, glUtils._textures[uid + "_colorLUT"]);
    gl.uniform1i(gl.getUniformLocation(program, "u_colorLUT"), 0);

    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, numEdges);

    // Restore render pipeline state
    gl.bindVertexArray(null);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.disable(gl.BLEND);
    gl.useProgram(null);
}


glUtils._drawPickingPass = function(gl, viewportTransform, markerScaleAdjusted) {
    // Set up render pipeline
    const program = glUtils._programs["picking"];
    gl.useProgram(program);

    // Set per-scene uniforms
    gl.uniformMatrix2fv(gl.getUniformLocation(program, "u_viewportTransform"), false, viewportTransform);
    gl.uniform2fv(gl.getUniformLocation(program, "u_canvasSize"), [gl.canvas.width, gl.canvas.height]);
    gl.uniform2fv(gl.getUniformLocation(program, "u_pickingLocation"), glUtils._pickingLocation);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale"), markerScaleAdjusted);
    gl.uniform1f(gl.getUniformLocation(program, "u_maxPointSize"), glUtils._useInstancing ? 2048 : 256);
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
        gl.uniform1f(gl.getUniformLocation(program, "u_transformIndex"),
            glUtils._collectionItemIndex[uid] != null ? glUtils._collectionItemIndex[uid] : -1);
        gl.uniform1f(gl.getUniformLocation(program, "u_globalMarkerScale"), glUtils._globalMarkerScale * glUtils._markerScaleFactor[uid]);
        gl.uniform1i(gl.getUniformLocation(program, "u_usePiechartFromMarker"), glUtils._usePiechartFromMarker[uid]);
        gl.uniform1i(gl.getUniformLocation(program, "u_useShapeFromMarker"), glUtils._useShapeFromMarker[uid]);
        gl.uniform1f(gl.getUniformLocation(program, "u_markerOpacity"), glUtils._markerOpacity[uid]);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, glUtils._textures[uid + "_colorLUT"]);
        gl.uniform1i(gl.getUniformLocation(program, "u_colorLUT"), 0);

        // 1st pass: clear the corner pixel
        gl.uniform1i(gl.getUniformLocation(program, "u_op"), 0);
        gl.drawArrays(gl.POINTS, 0, 1);  // Note: this drawcall does not have to be indexed
        // 2nd pass: draw all the markers (as single pixels)
        gl.uniform1i(gl.getUniformLocation(program, "u_op"), 1);
        gl.drawElements(gl.POINTS, numPoints, gl.UNSIGNED_INT, 0);

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
    glUtils._textures["shapeAtlas"] = glUtils._loadTextureFromImageURL(gl, glUtils._markershapes);
    glUtils._buffers["transformUBO"] = glUtils._createUniformBuffer(gl);

    // Restore per-markers WebGL objects
    for (let [uid, numPoints] of Object.entries(glUtils._numPoints)) {
        delete glUtils._buffers[uid + "_markers"];
        delete glUtils._vaos[uid + "_markers"];
        delete glUtils._textures[uid + "_colorLUT"];
        delete glUtils._textures[uid + "_colorscale"];
        glUtils.loadMarkers(uid);
    }

    glUtils.draw();  // Make sure markers are redrawn
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

    // Place marker canvas under the OSD canvas. Doing this also enables proper
    // compositing with the minimap and other OSD elements.
    const osd = document.getElementsByClassName("openseadragon-canvas")[0];
    osd.appendChild(canvas);

    this._programs["markers"] = this._loadShaderProgram(gl, this._markersVS, this._markersFS);
    this._programs["markers_instanced"] = this._loadShaderProgram(gl, this._markersVS, this._markersFS, "#define USE_INSTANCING\n");
    this._programs["picking"] = this._loadShaderProgram(gl, this._pickingVS, this._pickingFS);
    this._programs["edges"] = this._loadShaderProgram(gl, this._edgesVS, this._edgesFS);
    this._textures["shapeAtlas"] = this._loadTextureFromImageURL(gl, glUtils._markershapes);
    this._buffers["transformUBO"] = this._createUniformBuffer(gl);

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
