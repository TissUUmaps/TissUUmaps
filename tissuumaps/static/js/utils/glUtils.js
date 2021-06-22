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
    _markerScalarRange: [0.0, 1.0],
    _markerOpacity: 1.0,
    _useColorFromMarker: false,
    _usePiechartFromMarker: false,
    _colorscaleName: "null",
    _colorscaleData: [],
    _barcodeToLUTIndex: {},
    _barcodeToKey: {},
    _options: {antialias: false},
    _showColorbar: true,
    _piechartPalette: ["#d40328", "#22ac33", "#517bb1", "#f181af", "#ec9b05", "#b4d98b", "#9e4194", "#dc197c"]
}


glUtils._markersVS = `
    uniform vec2 u_imageSize;
    uniform vec4 u_viewportRect;
    uniform mat2 u_viewportTransform;
    uniform int u_markerType;
    uniform float u_markerScale;
    uniform vec2 u_markerScalarRange;
    uniform float u_markerOpacity;
    uniform bool u_useColorFromMarker;
    uniform bool u_usePiechartFromMarker;
    uniform bool u_alphaPass;
    uniform sampler2D u_colorLUT;
    uniform sampler2D u_colorscale;

    attribute vec4 a_position;

    varying vec4 v_color;
    varying vec2 v_shapeOrigin;
    varying float v_shapeSector;
    varying float v_shapeSize;

    #define MARKER_TYPE_BARCODE 0
    #define MARKER_TYPE_CP 1
    #define SHAPE_GRID_SIZE 4.0

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
            if (u_usePiechartFromMarker) {
                v_color.rgb = hex_to_rgb(a_position.w);
                v_color.a = 7.0 / 255.0;  // Give markers a round shape
                if (u_alphaPass) v_color.a *= float(a_position.z == 1.0);
            } else {
            v_color = texture2D(u_colorLUT, vec2(a_position.z, 0.5));
            }
        } else if (u_markerType == MARKER_TYPE_CP) {
            vec2 range = u_markerScalarRange;
            float normalized = (a_position.z - range[0]) / (range[1] - range[0]);
            v_color.rgb = texture2D(u_colorscale, vec2(normalized, 0.5)).rgb;
            v_color.a = 7.0 / 255.0;  // Give CP markers a round shape
        }

        if (u_useColorFromMarker) v_color.rgb = hex_to_rgb(a_position.w);

        gl_Position = vec4(ndcPos, 0.0, 1.0);
        gl_PointSize = max(2.0, u_markerScale / u_viewportRect.w);

        v_shapeOrigin.x = mod(v_color.a * 255.0 - 1.0, SHAPE_GRID_SIZE);
        v_shapeOrigin.y = floor((v_color.a * 255.0 - 1.0) / SHAPE_GRID_SIZE);
        v_shapeSector = a_position.z;  // TODO
        v_shapeSize = gl_PointSize;

        // Discard point here in vertex shader if marker is hidden
        v_color.a = v_color.a > 0.0 ? u_markerOpacity : 0.0;
        if (v_color.a == 0.0) gl_Position = vec4(2.0, 2.0, 2.0, 0.0);
    }
`;


glUtils._markersFS = `
    precision mediump float;

    uniform bool u_usePiechartFromMarker;
    uniform sampler2D u_shapeAtlas;

    varying vec4 v_color;
    varying vec2 v_shapeOrigin;
    varying float v_shapeSector;
    varying float v_shapeSize;

    #define UV_SCALE 0.7
    #define SHAPE_GRID_SIZE 4.0

    float sectorToAlpha(float sector, vec2 delta)
    {
        vec2 dir = normalize(gl_PointCoord.xy + delta - 0.5);
        float theta = atan(dir.x, dir.y);
        return float(theta < (sector * 2.0 - 1.0) * 3.141592);
    }

    float sectorToAlphaAA(float sector, float pointSize)
    {
        float accum = 0.0;
        accum += sectorToAlpha(sector, vec2(-0.25, -0.25) / pointSize);
        accum += sectorToAlpha(sector, vec2( 0.25, -0.25) / pointSize);
        accum += sectorToAlpha(sector, vec2(-0.25,  0.25) / pointSize);
        accum += sectorToAlpha(sector, vec2( 0.25,  0.25) / pointSize);
        return accum / 4.0;
    }

    void main()
    {
        vec2 uv = (gl_PointCoord.xy - 0.5) * UV_SCALE + 0.5;
        uv = (uv + v_shapeOrigin) * (1.0 / SHAPE_GRID_SIZE);

        vec4 shapeColor = texture2D(u_shapeAtlas, uv, -0.5);
        float shapeColorBias = max(0.0, 1.0 - v_shapeSize * 0.2);
        shapeColor.rgb = clamp(shapeColor.rgb + shapeColorBias, 0.0, 1.0);

        if (u_usePiechartFromMarker) {
            shapeColor.a *= sectorToAlphaAA(v_shapeSector, v_shapeSize);
        }

        gl_FragColor = shapeColor * v_color;
        gl_FragColor.rgb *= gl_FragColor.a;  // Need to pre-multiply alpha
        if (gl_FragColor.a < 0.01) discard;
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
    const positions = [];
    for (let i = 0; i < numPoints; ++i) {
        positions[4 * i + 0] = Math.random();  // X-coord
        positions[4 * i + 1] = Math.random();  // Y-coord
        positions[4 * i + 2] = Math.random();  // LUT-coord
        positions[4 * i + 3] = i / numPoints;  // Scalar data
    }

    const bytedata = new Float32Array(positions);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer); 
    gl.bufferData(gl.ARRAY_BUFFER, bytedata, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    return buffer;
}


// Generate a list of random normalized sector angles in format (TODO)
glUtils._createDummySectors = function(numSectors) {
    let sectors = [], sum = 0;
    for (let i = 0; i < numSectors; ++i) {
        sectors[i] = Math.floor(Math.random() * 100.0);
        sum += sectors[i];
    }
    for (let i = 0; i < numSectors; ++i) {
        sectors[i] /= sum;
    }
    for (let i = numSectors - 2; i >= 0; --i) {
        sectors[i] += sectors[i + 1];
    }
    return sectors;
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

    const sectorsPropertyName = markerUtils._uniquePiechartSelector;
    const usePiechartFromMarker = markerUtils._uniquePiechart && (sectorsPropertyName in markerData[0]);
    const piechartPalette = glUtils._piechartPalette;

    const positions = [];
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
                positions[4 * k + 2] = piechartAngles[j];
                positions[4 * k + 3] = Number("0x" + hexColor.substring(1,7));
            }
        }
        numPoints *= numSectors;
    } else {
    for (let i = 0; i < numPoints; ++i) {
        if (useColorFromMarker) hexColor = markerData[i][colorPropertyName];
        positions[4 * i + 0] = markerData[i].global_X_pos / imageWidth;
        positions[4 * i + 1] = markerData[i].global_Y_pos / imageHeight;
        positions[4 * i + 2] = glUtils._barcodeToLUTIndex[markerData[i].letters] / 4095.0;
        positions[4 * i + 3] = Number("0x" + hexColor.substring(1,7));
    }
    }

    const bytedata = new Float32Array(positions);

    gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers["barcodeMarkers"]);
    gl.bufferData(gl.ARRAY_BUFFER, bytedata, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    glUtils._numBarcodePoints = numPoints;
    glUtils._useColorFromMarker = useColorFromMarker;
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

    const positions = [];
    let scalarRange = [1e9, -1e9];
    for (let i = 0; i < numPoints; ++i) {
        if (useColorFromMarker) hexColor = markerData[i][propertyName];
        positions[4 * i + 0] = Number(markerData[i][xColumnName]) / imageWidth;
        positions[4 * i + 1] = Number(markerData[i][yColumnName]) / imageHeight;
        positions[4 * i + 2] = Number(markerData[i][propertyName]);
        positions[4 * i + 3] = Number("0x" + hexColor.substring(1,7));
        scalarRange[0] = Math.min(scalarRange[0], positions[4 * i + 2]);
        scalarRange[1] = Math.max(scalarRange[1], positions[4 * i + 2]);
    }

    const bytedata = new Float32Array(positions);

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
        const shape = document.getElementById(key + "-shape-ISS").value;
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
    image.src = "data:image/png;base64,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";

    return texture;
}


// @deprecated Not required anymore, but kept for backwards-compatibility
glUtils.clearNavigatorArea = function() {}


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

    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const program = glUtils._programs["markers"];

    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    const POSITION = gl.getAttribLocation(program, "a_position");
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

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    gl.bindBuffer(gl.ARRAY_BUFFER, glUtils._buffers["barcodeMarkers"]);
    gl.enableVertexAttribArray(POSITION);
    gl.vertexAttribPointer(POSITION, 4, gl.FLOAT, false, 0, 0);
    gl.uniform1i(gl.getUniformLocation(program, "u_markerType"), 0);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale"), glUtils._markerScale);
    gl.uniform1i(gl.getUniformLocation(program, "u_useColorFromMarker"), glUtils._useColorFromMarker);
    gl.uniform1i(gl.getUniformLocation(program, "u_usePiechartFromMarker"), glUtils._usePiechartFromMarker);
    if (glUtils._usePiechartFromMarker) {
        // 1st pass: draw alpha for whole marker shapes
        gl.uniform1i(gl.getUniformLocation(program, "u_alphaPass"), true);
        gl.colorMask(false, false, false, true);
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
    gl.uniform1i(gl.getUniformLocation(program, "u_markerType"), 1);
    gl.uniform1f(gl.getUniformLocation(program, "u_markerScale"), glUtils._markerScale * 0.5);
    gl.uniform1i(gl.getUniformLocation(program, "u_useColorFromMarker"),
        glUtils._colorscaleName.includes("ownColorFromColumn"));
    gl.uniform1i(gl.getUniformLocation(program, "u_usePiechartFromMarker"), false);
    if (glUtils._colorscaleName != "null") {  // Only show markers when a colorscale is selected
        gl.drawArrays(gl.POINTS, 0, glUtils._numCPPoints);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    gl.blendFunc(gl.ONE, gl.ONE);
    gl.disable(gl.BLEND);
    gl.useProgram(null);
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

    glUtils._initialized = true;
    glUtils.resize();  // Force initial resize to OSD canvas size
}
