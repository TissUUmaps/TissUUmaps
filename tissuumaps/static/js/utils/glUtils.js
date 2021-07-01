/**
* @file glUtils.js Utilities for WebGL-based marker drawing
* @author Fredrik Nysjo
* @see {@link glUtils}
*/
glUtils = {
    _initialized: false,
    _programs: {},
    _buffers: {},
    _textures: {},
    _numBarcodePoints: 0,
    _numCPPoints: 0,
    _imageSize: [1, 1],
    _viewportRect: [0, 0, 1, 1],
    _markerScale: 1.0,
    _useMarkerScaleFix: true,
    _markerScale2: 1.0,
    _markerScalarRange: [0.0, 1.0],
    _markerOpacity: 1.0,
    _useColorFromMarker: false,
    _useScaleFromMarker: false,
    _usePiechartFromMarker: false,
    _pickedMarker: -1,
    _pickingEnabled: false,
    _pickingLocation: [0.0, 0.0],
    _colorscaleName: "null",
    _colorscaleData: [],
    _barcodeToLUTIndex: {},
    _barcodeToKey: {},
    _options: {antialias: false, premultipliedAlpha: false},
    _showColorbar: true,
    _showMarkerInfo: true,
    _piechartPalette: ["#fff100", "#ff8c00", "#e81123", "#ec008c", "#68217a", "#00188f", "#00bcf2", "#00b294", "#009e49", "#bad80a"]
}


glUtils._markersVS = `
    uniform vec2 u_imageSize;
    uniform vec4 u_viewportRect;
    uniform mat2 u_viewportTransform;
    uniform int u_markerType;
    uniform float u_markerScale;
    uniform float u_markerScale2;
    uniform vec2 u_markerScalarRange;
    uniform float u_markerOpacity;
    uniform bool u_useColorFromMarker;
    uniform bool u_usePiechartFromMarker;
    uniform bool u_alphaPass;
    uniform float u_pickedMarker;
    uniform sampler2D u_colorLUT;
    uniform sampler2D u_colorscale;

    attribute vec4 a_position;
    attribute float a_index;
    attribute float a_scale;

    varying vec4 v_color;
    varying vec2 v_shapeOrigin;
    varying float v_shapeSector;
    varying float v_shapeSize;

    #define MARKER_TYPE_BARCODE 0
    #define MARKER_TYPE_CP 1
    #define SHAPE_GRID_SIZE 4.0
    #define DISCARD_VERTEX { gl_Position = vec4(2.0, 2.0, 2.0, 0.0); return; }

    vec3 hex_to_rgb(float v)
    {
        // Extract RGB color from 24-bit hex color stored in float
        v = clamp(v, 0.0, 16777215.0);
        return floor(mod((v + 0.49) / vec3(65536.0, 256.0, 1.0), 256.0)) / 255.0;
    }

    void main()
    {
        vec2 imagePos = a_position.xy * u_imageSize;
        vec2 viewportPos = imagePos - u_viewportRect.xy;
        vec2 ndcPos = (viewportPos / u_viewportRect.zw) * 2.0 - 1.0;
        ndcPos.y = -ndcPos.y;
        ndcPos = u_viewportTransform * ndcPos;

        if (u_markerType == MARKER_TYPE_BARCODE) {
            float barcodeID = mod(a_position.z, 4096.0);
            v_color = texture2D(u_colorLUT, vec2(barcodeID / 4095.0, 0.5));

            if (u_usePiechartFromMarker && v_color.a > 0.0) {
                v_shapeSector = a_position.z / 16777215.0;
                v_color.rgb = hex_to_rgb(a_position.w);
                v_color.a = 8.0 / 255.0;  // Give markers a round shape
                if (u_pickedMarker == a_index) v_color.a = 7.0 / 255.0;
                if (u_alphaPass) v_color.a *= float(v_shapeSector > 0.999);
            }
        } else if (u_markerType == MARKER_TYPE_CP) {
            vec2 range = u_markerScalarRange;
            float normalized = (a_position.z - range[0]) / (range[1] - range[0]);
            v_color.rgb = texture2D(u_colorscale, vec2(normalized, 0.5)).rgb;
            v_color.a = 7.0 / 255.0;  // Give CP markers a round shape
        }

        if (u_useColorFromMarker) v_color.rgb = hex_to_rgb(a_position.w);

        gl_Position = vec4(ndcPos, 0.0, 1.0);
        gl_PointSize = max(2.0, min(256.0, a_scale * u_markerScale * u_markerScale2 / u_viewportRect.w));

        v_shapeOrigin.x = mod((v_color.a + 0.00001) * 255.0 - 1.0, SHAPE_GRID_SIZE);
        v_shapeOrigin.y = floor(((v_color.a + 0.00001) * 255.0 - 1.0) / SHAPE_GRID_SIZE);
        v_shapeSize = gl_PointSize;

        // Discard point here in vertex shader if marker is hidden
        v_color.a = v_color.a > 0.0 ? u_markerOpacity : 0.0;
        if (v_color.a == 0.0) DISCARD_VERTEX;
    }
`;


glUtils._markersFS = `
    precision mediump float;

    uniform bool u_usePiechartFromMarker;
    uniform bool u_alphaPass;
    uniform sampler2D u_shapeAtlas;

    varying vec4 v_color;
    varying vec2 v_shapeOrigin;
    varying float v_shapeSector;
    varying float v_shapeSize;

    #define UV_SCALE 0.7
    #define SHAPE_GRID_SIZE 4.0

    float sectorToAlpha(float sector, vec2 uv)
    {
        vec2 dir = normalize(uv - 0.5);
        float theta = atan(dir.x, dir.y);
        return float(theta < (sector * 2.0 - 1.0) * 3.141592);
    }

    float sectorToAlphaAA(float sector, vec2 uv, float delta)
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
        vec2 uv = (gl_PointCoord.xy - 0.5) * UV_SCALE + 0.5;
        uv = (uv + v_shapeOrigin) * (1.0 / SHAPE_GRID_SIZE);

        vec4 shapeColor = texture2D(u_shapeAtlas, uv, -0.5);
        float shapeColorBias = max(0.0, 1.0 - v_shapeSize * 0.2);
        shapeColor.rgb = clamp(shapeColor.rgb + shapeColorBias, 0.0, 1.0);

        if (u_usePiechartFromMarker && !u_alphaPass) {
            float delta = 0.25 / v_shapeSize;
            shapeColor.a *= sectorToAlphaAA(v_shapeSector, gl_PointCoord, delta);
        }

        gl_FragColor = shapeColor * v_color;
        if (gl_FragColor.a < 0.01) discard;
    }
`;


glUtils._pickingVS = `
    uniform vec2 u_imageSize;
    uniform vec4 u_viewportRect;
    uniform mat2 u_viewportTransform;
    uniform vec2 u_canvasSize;
    uniform vec2 u_pickingLocation;
    uniform float u_markerScale;
    uniform float u_markerScale2;
    uniform int u_op;
    uniform sampler2D u_colorLUT;

    attribute vec4 a_position;
    attribute float a_index;
    attribute float a_scale;

    varying vec4 v_color;

    #define OP_CLEAR 0
    #define OP_WRITE_INDEX 1
    #define DISCARD_VERTEX { gl_Position = vec4(2.0, 2.0, 2.0, 0.0); return; }

    vec3 hex_to_rgb(float v)
    {
        // Extract RGB color from 24-bit hex color stored in float
        v = clamp(v, 0.0, 16777215.0);
        return floor(mod((v + 0.49) / vec3(65536.0, 256.0, 1.0), 256.0)) / 255.0;
    }

    void main()
    {
        vec2 imagePos = a_position.xy * u_imageSize;
        vec2 viewportPos = imagePos - u_viewportRect.xy;
        vec2 ndcPos = (viewportPos / u_viewportRect.zw) * 2.0 - 1.0;
        ndcPos.y = -ndcPos.y;
        ndcPos = u_viewportTransform * ndcPos;

        v_color = vec4(0.0);
        if (u_op == OP_WRITE_INDEX) {
            float barcodeID = mod(a_position.z, 4096.0);
            float shapeID = texture2D(u_colorLUT, vec2(barcodeID / 4095.0, 0.5)).a;
            if (shapeID == 0.0) DISCARD_VERTEX;

            vec2 canvasPos = (ndcPos * 0.5 + 0.5) * u_canvasSize;
            canvasPos.y = (u_canvasSize.y - canvasPos.y);  // Y-axis is inverted
            float pointSize = max(2.0, min(256.0, a_scale * u_markerScale * u_markerScale2 / u_viewportRect.w));
            
            // TODO This test works as an inside/outside test for the special
            // case where the marker shape is round; for the general case, we
            // would need to sample the shape texture of each marker.
            if (length(canvasPos - u_pickingLocation) > pointSize * 0.4) DISCARD_VERTEX;

            v_color.rgb = hex_to_rgb(a_index + float(u_op));
        }

        gl_Position = vec4(-0.9999, -0.9999, 0.0, 1.0);
        gl_PointSize = 1.0;
    }
`;


glUtils._pickingFS = `
    precision mediump float;

    varying vec4 v_color;

    void main()
    {
        gl_FragColor = v_color;
    }
`;


glUtils._loadShaderProgram = function(gl, vertSource, fragSource) {
    const vertShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertShader, vertSource);
    gl.compileShader(vertShader);
    if (!gl.getShaderParameter(vertShader, gl.COMPILE_STATUS)) {
        console.log("Could not compile vertex shader: " + gl.getShaderInfoLog(vertShader));
    }

    const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragShader, fragSource);
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


glUtils._createDummyMarkerBuffer = function(gl, numPoints) {
    const positions = [], indices = [], scales = [];
    for (let i = 0; i < numPoints; ++i) {
        positions[4 * i + 0] = Math.random();  // X-coord
        positions[4 * i + 1] = Math.random();  // Y-coord
        positions[4 * i + 2] = Math.random();  // LUT-coord
        positions[4 * i + 3] = i / numPoints;  // Scalar data
        indices[i] = i;  // Store index needed for picking
        scales[i] = 1.0;  // Marker scale factor
    }

    const bytedata = new Float32Array(positions.concat(indices.concat(scales)));

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer); 
    gl.bufferData(gl.ARRAY_BUFFER, bytedata, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

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


// Load barcode markers loaded from CSV file into vertex buffer
glUtils.loadMarkers = function() {
    if (!glUtils._initialized) return;
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl", glUtils._options);

    const markerData = dataUtils["ISS_processeddata"];
    let numPoints = markerData.length;
    const keyName = document.getElementById("ISS_key_header").value;
    const imageWidth = OSDViewerUtils.getImageWidth();
    const imageHeight = OSDViewerUtils.getImageHeight();

    // If new marker data was loaded, we need to assign each barcode an index
    // that we can use with the LUT textures for color, visibility, etc.
    glUtils._updateBarcodeToLUTIndexDict(markerData, keyName);

    const colorPropertyName = markerUtils._uniqueColorSelector;
    const useColorFromMarker = markerUtils._uniqueColor && (colorPropertyName in markerData[0]);
    let hexColor = "#000000";

    const scalePropertyName = markerUtils._uniqueScaleSelector;
    const useScaleFromMarker = markerUtils._uniqueScale && (scalePropertyName in markerData[0]);

    const sectorsPropertyName = markerUtils._uniquePiechartSelector;
    const usePiechartFromMarker = markerUtils._uniquePiechart && (sectorsPropertyName in markerData[0]);
    const piechartPalette = glUtils._piechartPalette;

    const positions = [], indices = [], scales = [];
    if (usePiechartFromMarker) {
        const numSectors = markerData[0][sectorsPropertyName].split(";").length;
        for (let i = 0; i < numPoints; ++i) {
            const sectors = markerData[i][sectorsPropertyName].split(";");
            const piechartAngles = glUtils._createPiechartAngles(sectors);
            for (let j = 0; j < numSectors; ++j) {
                const k = (i * numSectors + j);
                hexColor = piechartPalette[j % piechartPalette.length];
                positions[4 * k + 0] = markerData[i].global_X_pos / imageWidth;
                positions[4 * k + 1] = markerData[i].global_Y_pos / imageHeight;
                positions[4 * k + 2] = glUtils._barcodeToLUTIndex[markerData[i].letters] +
                                       Math.floor(piechartAngles[j] * 4095.0) * 4096.0;
                positions[4 * k + 3] = Number("0x" + hexColor.substring(1,7));
                indices[k] = i;  // Store index needed for picking
                if (useScaleFromMarker) scales[k] = markerData[i][scalePropertyName];
                else scales[k] = 1.0;  // Marker scale factor
            }
        }
        numPoints *= numSectors;
    } else {
        for (let i = 0; i < numPoints; ++i) {
            if (useColorFromMarker) hexColor = markerData[i][colorPropertyName];
            positions[4 * i + 0] = markerData[i].global_X_pos / imageWidth;
            positions[4 * i + 1] = markerData[i].global_Y_pos / imageHeight;
            positions[4 * i + 2] = glUtils._barcodeToLUTIndex[markerData[i].letters];
            positions[4 * i + 3] = Number("0x" + hexColor.substring(1,7));
            indices[i] = i;  // Store index needed for picking
            if (useScaleFromMarker) scales[i] = markerData[i][scalePropertyName];
            else scales[i] = 1.0;  // Marker scale factor
        }
    }

    const bytedata = new Float32Array(positions.concat(indices.concat(scales)));

    gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers["barcodeMarkers"]);
    gl.bufferData(gl.ARRAY_BUFFER, bytedata, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    glUtils._numBarcodePoints = numPoints;
    glUtils._useColorFromMarker = useColorFromMarker;
    glUtils._useScaleFromMarker = useScaleFromMarker;
    glUtils._usePiechartFromMarker = usePiechartFromMarker;
    glUtils.updateLUTTextures();
}


// Load cell morphology markers loaded from CSV file into vertex buffer
glUtils.loadCPMarkers = function() {
    if (!glUtils._initialized) return;
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl", glUtils._options);

    const markerData = CPDataUtils["CP_rawdata"];
    const numPoints = markerData.length;
    const propertyName = document.getElementById("CP_property_header").value;
    const xColumnName = document.getElementById("CP_X_header").value;
    const yColumnName = document.getElementById("CP_Y_header").value;
    const colorscaleName = document.getElementById("CP_colorscale").value;
    const imageWidth = OSDViewerUtils.getImageWidth();
    const imageHeight = OSDViewerUtils.getImageHeight();

    const useColorFromMarker = colorscaleName.includes("ownColorFromColumn");
    let hexColor = "#000000";

    const scalePropertyName = markerUtils._uniqueScaleSelector;
    const useScaleFromMarker = markerUtils._uniqueScale && (scalePropertyName in markerData[0]);

    const positions = [], indices = [], scales = [];
    let scalarRange = [1e9, -1e9];
    for (let i = 0; i < numPoints; ++i) {
        if (useColorFromMarker) hexColor = markerData[i][propertyName];
        positions[4 * i + 0] = Number(markerData[i][xColumnName]) / imageWidth;
        positions[4 * i + 1] = Number(markerData[i][yColumnName]) / imageHeight;
        positions[4 * i + 2] = Number(markerData[i][propertyName]);
        positions[4 * i + 3] = Number("0x" + hexColor.substring(1,7));
        indices[i] = i;  // Store index needed for picking
        if (useScaleFromMarker) scales[i] = markerData[i][scalePropertyName];
        else scales[i] = 1.0;

        scalarRange[0] = Math.min(scalarRange[0], positions[4 * i + 2]);
        scalarRange[1] = Math.max(scalarRange[1], positions[4 * i + 2]);
    }

    const bytedata = new Float32Array(positions.concat(indices.concat(scales)));

    gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers["CPMarkers"]);
    gl.bufferData(gl.ARRAY_BUFFER, bytedata, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    glUtils._numCPPoints = numPoints;
    glUtils._markerScalarRange = scalarRange;
    glUtils._colorscaleName = colorscaleName;
    glUtils._updateColorScaleTexture(gl, glUtils._textures["colorscale"]);
    glUtils._updateColorbarCanvas(colorscaleName, glUtils._colorscaleData, propertyName, scalarRange);
    glUtils.draw();  // Force redraw
}


glUtils._updateBarcodeToLUTIndexDict = function(markerData, keyName) {
    const barcodeToLUTIndex = {};
    const barcodeToKey = {};
    const numPoints = markerData.length;
    for (let i = 0, index = 0; i < numPoints; ++i) {
        const barcode = markerData[i].letters;
        const gene_name = markerData[i].gene_name;
        if (!(barcode in barcodeToLUTIndex)) {
            barcodeToLUTIndex[barcode] = index++;
            barcodeToKey[barcode] = (keyName == "letters" ? barcode : gene_name);
        }
    }
    glUtils._barcodeToLUTIndex = barcodeToLUTIndex;
    glUtils._barcodeToKey = barcodeToKey;
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


glUtils._updateColorLUTTexture = function(gl, texture) {
    const allMarkersCheckbox = document.getElementById("AllMarkers-checkbox-ISS");
    const showAll = allMarkersCheckbox && allMarkersCheckbox.checked;

    const colors = new Array(4096 * 4);
    for (let [barcode, index] of Object.entries(glUtils._barcodeToLUTIndex)) {
        // Get color, shape, etc. from HTML input elements for barcode
        const key = glUtils._barcodeToKey[barcode];  // Could be barcode or gene name
        hexInput = document.getElementById(key + "-color-ISS")
        if (hexInput) {
            var hexColor = document.getElementById(key + "-color-ISS").value;
        }
        else {
            var hexColor = "#000000";
        }
        shapeInput = document.getElementById(key + "-shape-ISS")
        if (shapeInput) {
            var shape = document.getElementById(key + "-shape-ISS").value;
        }
        else {
            var shape = "";
        };
        const visible = showAll || markerUtils._checkBoxes[key].checked;
        colors[4 * index + 0] = Number("0x" + hexColor.substring(1,3)); 
        colors[4 * index + 1] = Number("0x" + hexColor.substring(3,5));
        colors[4 * index + 2] = Number("0x" + hexColor.substring(5,7));
        colors[4 * index + 3] = Number(visible) * (Number(shape) + 1);
    }

    const bytedata = new Uint8Array(colors);

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 4096, 1, 0, gl.RGBA,
                  gl.UNSIGNED_BYTE, bytedata);
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


glUtils._updateColorScaleTexture = function(gl, texture) {
    const colors = [];
    for (let i = 0; i < 256; ++i) {
        const normalized = i / 255.0;
        if (glUtils._colorscaleName.includes("interpolate") &&
            !glUtils._colorscaleName.includes("Rainbow")) {
            const color = d3[glUtils._colorscaleName](normalized);
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
    glUtils._colorscaleData = colors;

    const bytedata = new Uint8Array(colors);

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA,
                  gl.UNSIGNED_BYTE, bytedata);
    gl.bindTexture(gl.TEXTURE_2D, null);
}


glUtils._updateColorbarCanvas = function(colorscaleName, colorscaleData, propertyName, propertyRange) {
    const canvas = document.getElementById("CP_colorbar");
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    if (!glUtils._showColorbar || colorscaleName == "null" ||
        colorscaleName == "ownColorFromColumn") return;

    const gradient = ctx.createLinearGradient(64, 0, 256+64, 0);
    const numStops = 32;
    for (let i = 0; i < numStops; ++i) {
        const normalized = i / (numStops - 1);
        const index = Math.floor(normalized * 255.99);
        const r = Math.floor(colorscaleData[4 * index + 0]);
        const g = Math.floor(colorscaleData[4 * index + 1]);
        const b = Math.floor(colorscaleData[4 * index + 2]);
        gradient.addColorStop(normalized, "rgb(" + r + "," + g + "," + b + ")");
    }

    // Draw colorbar (with outline)
    ctx.fillStyle = gradient;
    ctx.fillRect(64, 64, 256, 16);
    ctx.strokeStyle = "#555";
    ctx.strokeRect(64, 64, 256, 16);

    // Convert range annotations to scientific notation if they may overflow
    let propertyMin = propertyRange[0].toString();
    let propertyMax = propertyRange[1].toString();
    if (propertyMin.length > 9) propertyMin = propertyRange[0].toExponential(5);
    if (propertyMax.length > 9) propertyMax = propertyRange[1].toExponential(5);

    // Draw annotations (with drop shadow)
    ctx.font = "16px Arial";
    ctx.textAlign = "center";
    ctx.fillStyle = "#000";  // Shadow color
    ctx.fillText(propertyName, ctx.canvas.width/2+1, 32+1);
    ctx.fillText(propertyMin, ctx.canvas.width/2-128+1, 56+1);
    ctx.fillText(propertyMax, ctx.canvas.width/2+128+1, 56+1);
    ctx.fillStyle = "#fff";  // Text color
    ctx.fillText(propertyName, ctx.canvas.width/2, 32);
    ctx.fillText(propertyMin, ctx.canvas.width/2-128, 56);
    ctx.fillText(propertyMax, ctx.canvas.width/2+128, 56);
}


// Creates a 2D-canvas for drawing the colorbar on top of the WebGL-canvas
glUtils._createColorbarCanvas = function() {
    const root = document.getElementById("gl_canvas").parentElement;
    const canvas = document.createElement("canvas");
    root.appendChild(canvas);

    canvas.id = "CP_colorbar";
    canvas.width = "384";  // Fixed width in pixels
    canvas.height = "96";  // Fixed height in pixels
    canvas.style = "position:relative; float:left; width:31%; left:68%; " +
                   "margin-top:-9%; z-index:20; pointer-events:none";
}


// Creates WebGL canvas for drawing the markers
glUtils._createMarkerWebGLCanvas = function() {
    const canvas = document.createElement("canvas");
    canvas.id = "gl_canvas";
    canvas.width = "1"; canvas.height = "1";
    canvas.style = "position:relative; pointer-events:none";
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
    };
    image.src = src;
    return texture;
}


// @deprecated Not required anymore, but kept for backwards-compatibility
glUtils.clearNavigatorArea = function() {}


glUtils.drawColorPass = function(gl, viewportTransform, markerScaleAdjusted) {
    // Set up render pipeline
    const program = glUtils._programs["markers"];
    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const POSITION = gl.getAttribLocation(program, "a_position");
    const INDEX = gl.getAttribLocation(program, "a_index");
    const SCALE = gl.getAttribLocation(program, "a_scale");
    gl.uniform2fv(gl.getUniformLocation(program, "u_imageSize"), glUtils._imageSize);
    gl.uniform4fv(gl.getUniformLocation(program, "u_viewportRect"), glUtils._viewportRect);
    gl.uniformMatrix2fv(gl.getUniformLocation(program, "u_viewportTransform"), false, viewportTransform);
    gl.uniform2fv(gl.getUniformLocation(program, "u_markerScalarRange"), glUtils._markerScalarRange);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerOpacity"), glUtils._markerOpacity);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, glUtils._textures["colorLUT"]);
    gl.uniform1i(gl.getUniformLocation(program, "u_colorLUT"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, glUtils._textures["colorscale"]);
    gl.uniform1i(gl.getUniformLocation(program, "u_colorscale"), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, glUtils._textures["shapeAtlas"]);
    gl.uniform1i(gl.getUniformLocation(program, "u_shapeAtlas"), 2);

    gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers["barcodeMarkers"]);
    gl.enableVertexAttribArray(POSITION);
    gl.vertexAttribPointer(POSITION, 4, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(INDEX);
    gl.vertexAttribPointer(INDEX, 1, gl.FLOAT, false, 0, glUtils._numBarcodePoints * 16);
    gl.enableVertexAttribArray(SCALE);
    gl.vertexAttribPointer(SCALE, 1, gl.FLOAT, false, 0, glUtils._numBarcodePoints * 20);
    gl.uniform1i(gl.getUniformLocation(program, "u_markerType"), 0);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale"), markerScaleAdjusted);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale2"), glUtils._markerScale2);
    gl.uniform1i(gl.getUniformLocation(program, "u_useColorFromMarker"), glUtils._useColorFromMarker);
    gl.uniform1i(gl.getUniformLocation(program, "u_usePiechartFromMarker"), glUtils._usePiechartFromMarker);
    gl.uniform1f(gl.getUniformLocation(program, "u_pickedMarker"), glUtils._pickedMarker);
    if (glUtils._usePiechartFromMarker) {
        // 1st pass: draw alpha for whole marker shapes
        gl.uniform1i(gl.getUniformLocation(program, "u_alphaPass"), true);
        gl.drawArrays(gl.POINTS, 0, glUtils._numBarcodePoints);
        // 2nd pass: draw colors for individual piechart sectors
        gl.uniform1i(gl.getUniformLocation(program, "u_alphaPass"), false);
        gl.colorMask(true, true, true, false);
        gl.drawArrays(gl.POINTS, 0, glUtils._numBarcodePoints);
        gl.colorMask(true, true, true, true);
    } else {
        gl.drawArrays(gl.POINTS, 0, glUtils._numBarcodePoints);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers["CPMarkers"]);
    gl.enableVertexAttribArray(POSITION);
    gl.vertexAttribPointer(POSITION, 4, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(INDEX);
    gl.vertexAttribPointer(INDEX, 1, gl.FLOAT, false, 0, glUtils._numCPPoints * 16);
    gl.enableVertexAttribArray(SCALE);
    gl.vertexAttribPointer(SCALE, 1, gl.FLOAT, false, 0, glUtils._numCPPoints * 20);
    gl.uniform1i(gl.getUniformLocation(program, "u_markerType"), 1);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale"), markerScaleAdjusted);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale2"), glUtils._markerScale2);
    gl.uniform1i(gl.getUniformLocation(program, "u_useColorFromMarker"),
        glUtils._colorscaleName.includes("ownColorFromColumn"));
    gl.uniform1i(gl.getUniformLocation(program, "u_usePiechartFromMarker"), false);
    if (glUtils._colorscaleName != "null") {  // Only show markers when a colorscale is selected
        gl.drawArrays(gl.POINTS, 0, glUtils._numCPPoints);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Restore render pipeline state
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.disable(gl.BLEND);
    gl.useProgram(null);
}


glUtils.drawPickingPass = function(gl, viewportTransform, markerScaleAdjusted) {
    if (!glUtils._usePiechartFromMarker) {
        glUtils._pickedMarker = -1;
        return;  // TODO: Right now, we only perform picking for piecharts
    }

    // Set up render pipeline
    const program = glUtils._programs["picking"];
    gl.useProgram(program);

    const POSITION = gl.getAttribLocation(program, "a_position");
    const INDEX = gl.getAttribLocation(program, "a_index");
    const SCALE = gl.getAttribLocation(program, "a_scale");
    gl.uniform2fv(gl.getUniformLocation(program, "u_imageSize"), glUtils._imageSize);
    gl.uniform4fv(gl.getUniformLocation(program, "u_viewportRect"), glUtils._viewportRect);
    gl.uniformMatrix2fv(gl.getUniformLocation(program, "u_viewportTransform"), false, viewportTransform);
    gl.uniform2fv(gl.getUniformLocation(program, "u_canvasSize"), [gl.canvas.width, gl.canvas.height]);
    gl.uniform2fv(gl.getUniformLocation(program, "u_pickingLocation"), glUtils._pickingLocation);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale2"), glUtils._markerScale2);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale"), markerScaleAdjusted);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, glUtils._textures["colorLUT"]);
    gl.uniform1i(gl.getUniformLocation(program, "u_colorLUT"), 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers["barcodeMarkers"]);
    gl.enableVertexAttribArray(POSITION);
    gl.vertexAttribPointer(POSITION, 4, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(INDEX);
    gl.vertexAttribPointer(INDEX, 1, gl.FLOAT, false, 0, glUtils._numBarcodePoints * 16);
    gl.enableVertexAttribArray(SCALE);
    gl.vertexAttribPointer(SCALE, 1, gl.FLOAT, false, 0, glUtils._numBarcodePoints * 20);
    // 1st pass: clear the corner pixel
    gl.uniform1i(gl.getUniformLocation(program, "u_op"), 0);
    gl.drawArrays(gl.POINTS, 0, 1);
    // 2nd pass: draw all the markers (as single pixels)
    gl.uniform1i(gl.getUniformLocation(program, "u_op"), 1);
    gl.drawArrays(gl.POINTS, 0, glUtils._numBarcodePoints);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Read back pixel at location (0, 0) to get the picked object
    const result = new Uint8Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, result);
    const picked = Number(result[2] + result[1] * 256 + result[0] * 65536) - 1;
    glUtils._pickedMarker = picked;

    // Restore render pipeline state
    gl.useProgram(null);
}


glUtils.draw = function() {
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl", glUtils._options);

    const bounds = tmapp["ISS_viewer"].viewport.getBounds();
    glUtils._viewportRect = [bounds.x, bounds.y, bounds.width, bounds.height];
    const homeBounds = tmapp["ISS_viewer"].world.getHomeBounds();
    glUtils._imageSize = [homeBounds.width, homeBounds.height];
    const orientationDegrees = tmapp["ISS_viewer"].viewport.getRotation();

    // The OSD viewer can be rotated, so need to apply the same transform to markers
    const t = orientationDegrees * (3.141592 / 180.0);
    const viewportTransform = [Math.cos(t), -Math.sin(t), Math.sin(t), Math.cos(t)];

    // Compute adjusted marker scale so that the actual marker size becomes less
    // dependant on screen resolution or window size
    let markerScaleAdjusted = glUtils._markerScale;
    if (glUtils._useMarkerScaleFix) markerScaleAdjusted *= (gl.canvas.height / 900.0);

    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    if (glUtils._pickingEnabled) {
        glUtils.drawPickingPass(gl, viewportTransform, markerScaleAdjusted);
        glUtils._pickingEnabled = false;  // Clear flag until next click event
    }

    glUtils.drawColorPass(gl, viewportTransform, markerScaleAdjusted);
}


glUtils.pick = function(event) {
    if (event.quick) {
        glUtils._pickingEnabled = true;
        glUtils._pickingLocation = [event.position.x, event.position.y];
        glUtils.draw();

        tmapp["ISS_viewer"].removeOverlay("ISS_marker_info");
        if (glUtils._pickedMarker >= 0 && glUtils._showMarkerInfo) {
            const div = document.createElement("div");
            div.id = "ISS_marker_info";
            div.width = "1px"; div.height = "1px";
            div.style = "background-color:white; margin:0px; padding:2px 6px; " +
                        "border:1px solid; z-index:10; opacity:80%; pointer-events:none";
            div.innerHTML = markerUtils.makePiechartTable(dataUtils["ISS_processeddata"][glUtils._pickedMarker]);

            tmapp["ISS_viewer"].addOverlay({
                element: div,
                placement: "TOP_LEFT",
                location: tmapp["ISS_viewer"].viewport.viewerElementToViewportCoordinates(event.position),
                checkResize: false,
                rotationMode: OpenSeadragon.OverlayRotationMode.NO_ROTATION
            });
        }
    }
}


glUtils.resize = function() {
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl", glUtils._options);

    const op = tmapp["object_prefix"];
    const newSize = tmapp[op + "_viewer"].viewport.containerSize;
    gl.canvas.width = newSize.x;
    gl.canvas.height = newSize.y;
}


glUtils.resizeAndDraw = function() {
    glUtils.resize();
    glUtils.draw();
}


// @deprecated Not required anymore, but kept for backwards-compatibility
glUtils.postRedraw = function() {}


glUtils.updateMarkerScale = function() {
    const globalMarkerSize = Number(document.getElementById("ISS_globalmarkersize_text").value);
    // Clamp the scale factor to avoid giant markers and slow rendering if the
    // user inputs a very large value (say 10000 or something)
    glUtils._markerScale = Math.max(0.01, Math.min(5.0, globalMarkerSize / 100.0));
}


glUtils.updateLUTTextures = function() {
    const canvas = document.getElementById("gl_canvas");
    const gl = canvas.getContext("webgl", glUtils._options);

    if (glUtils._numBarcodePoints > 0) {  // LUTs are currently only used for barcode data
        glUtils._updateColorLUTTexture(gl, glUtils._textures["colorLUT"]);
    }
}


glUtils.init = function() {
    if (glUtils._initialized) return;

    let canvas = document.getElementById("gl_canvas");
    if (!canvas) canvas = this._createMarkerWebGLCanvas();
    const gl = canvas.getContext("webgl", glUtils._options);

    // Place marker canvas under the OSD canvas. Doing this also enables proper
    // compositing with the minimap and other OSD elements.
    const osd = document.getElementsByClassName("openseadragon-canvas")[0];
    osd.appendChild(canvas);

    this._programs["markers"] = this._loadShaderProgram(gl, this._markersVS, this._markersFS);
    this._programs["picking"] = this._loadShaderProgram(gl, this._pickingVS, this._pickingFS);
    this._buffers["barcodeMarkers"] = this._createDummyMarkerBuffer(gl, this._numBarcodePoints);
    this._buffers["CPMarkers"] = this._createDummyMarkerBuffer(gl, this._numCPMarkers);
    this._textures["colorLUT"] = this._createColorLUTTexture(gl);
    this._textures["colorscale"] = this._createColorScaleTexture(gl);
    this._textures["shapeAtlas"] = this._loadTextureFromImageURL(gl, "/static/misc/markershapes.png");

    this._createColorbarCanvas();  // The colorbar is drawn separately in a 2D-canvas

    glUtils.updateMarkerScale();
    document.getElementById("ISS_globalmarkersize_text").addEventListener("input", glUtils.updateMarkerScale);
    document.getElementById("ISS_globalmarkersize_text").addEventListener("input", glUtils.draw);
    document.getElementById("ISS_markers").addEventListener("change", glUtils.updateLUTTextures);
    document.getElementById("ISS_markers").addEventListener("change", glUtils.draw);

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
