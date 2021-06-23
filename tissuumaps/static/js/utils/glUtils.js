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
    _piechartPalette: ["#fff100", "#ff8c00", "#e81123", "#ec008c", "#68217a", "#00188f", "#00bcf2", "#00b294", "#009e49", "#bad80a"]
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
                v_color.a = 8.0 / 255.0;  // Give markers a round shape
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
    image.src = src;"data:image/png;base64,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Vf/V7uhf//pH//rXP/pH/wYA41xZlo1arfbvkww2r991113ZtWuXasaJXbt25c4772xdHqzVateUZdmwQ/rXv/7Rv/71j/71r/+JoG2iH477779/c09PT1uS7lfXhoeHs3nz5sydOzdFUShoDKuqKt/85jezdevW1rt6Fy1adIcd0r/+9Y/+9a9/9K9//U8UNUckqarqxrS8FOjpp59OX1+fzRnjfvSjH+W5555rXX6oq6vrZruD/vWP/vWvf/Svf/0bAEwwZVkOt7W1fTbJfq/7efDBB7N69WobNEY99dRTWbZsWevyzkmTJn12wYIFdTuE/vWP/vWvf/Svf/0bAExACxcufCbJ51vXv/vd72bLli02aIzZvHlz7rjjda/wqYqiuOZLX/rSc3YI/esf9K9/9K9//U80bbbgf+vr61vV3d19dJL3v7pWr9ezZs2azJkzJx0dHTZpDHjxxRfzta99baTvdP1PZVn+rR1C//oH/esf9K//icgrAF7vPyZ5sHlhx44due222zI0NGR3RrnBwcHceuut2blzZ+td93V1dV1vh9C//kH/+gf9698AgCQvvx8oySeTPN28vmHDhtx+++2p1711ZLQaHh7O7bffno0bN7betSbJp73vB/3rH/Svf9C//icybwEYQV9f357LLrvs7qqqPpNk2qvr27dvz6ZNm9LV1eXrQUaZRqORpUuXZu3ata13bUtyWVmW6+0S+tc/6F//oH/9GwDwOvfff//27u7u5Un+KMmk107Ttm3Ztm1bzj33XBeBURT/t7/97Tz55JOtd+2t1WpXLlq06HG7hP71D/rXP+hf/wYAvKG+vr7nuru7VyS5qnmvtmzZkueffz6zZ89OreZdFIdTvV7Pt771rZG+rmUoyVWLFi26zy6hf/2D/vUP+tc/BgBv5SLwVE9Pz9okH0/TZyZs27YtGzZsyOzZs9PWZhsPh8HBwdx2220jveynnuTqsiy/Y5fQv/5B//oH/esfA4CDuQis6unpWZ/kw80Xge3bt+eXv/xlzjnnHF8R8i7bvXt3vvGNb2T9+te9tadeFMU1ZVneapfQv/5B//oH/esfA4C3cxF4vKen56kkH2vet927d2fVqlWZNWtWjjjiCBv1Lti8eXO+/vWvZ+vWra13DRdF8W8XLVr0dbuE/vUP+tc/6F//GAC8k4tAf09Pz4okn0jT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++yz86lPfSqTJk2ySYfr8La15bzzzsuWLVuydevW1ovAx3t6elb39fU9YafQv/5B//oH/esfA4A3i//SJP+clvf8zJkzJ1dddVXa2mzf4Var1TJ79uxs27YtW7Zsab0IfLS7u/vBvr6+5+wU+tc/6F//oH/9GwAwoiVLlsyuqureJPt9mee5556bP/zDP0ytVrNJo0RRFJk9e3a2bNnSehGYlORjl1122Xfvv//+bXYK/esf9K9/0L/+JzKneAQ33njj8fV6/ftJjmleP+uss3LVVVeJf5ReBP7gD/4gZ5xxRutdx9br9btvuummmXYJ/esf9K9/0L/+DQB4zS233NI+NDT0rSSnNa+fdNJJWbBggZf9jGJtbW351Kc+lRNPPLH1rjMGBweXlmXpDVvoX/+gf/2D/vVvAMDLNm7c+P8kuaR5bfr06fn0pz/tAz/GgPb29lx99dU55phjWu/qTnKzHUL/+gf96x/0r/+JyjiryeLFi69O8tfNa1OmTMnnPvc53/M5xi4Cp59+elasWJF6vd581/u7u7tX9/X19dsl9K9/0L/+Qf/6NwCYoMqyPD3JnUkmv7pWFEUWLFiQk08+2QaNMdOmTcvMmTOzatXrvgr097q7u7/Z19e3wy6hf/2D/vWP/vWv/4nEWwBejn9Skm8kOap5/Xd+53dy1lln2aAx6pxzzskll1zSunx0km8sXbrU8Av96x/961//6F//+jcAmIAWJnl/88IZZ5yRD37wg3ZmjLv00ktz2mmntS5fvHr16r+wO+hf/+hf//pH//rXvwHABFKW5dwkX2xe6+zszMc//vEUReGEjHGvfj3I1KlT91uvqupLN9xww3l2SP/61z/617/+0b/+9W8AMDHiryX5uyTtzeu///u/nyOOOEI948SRRx6Zj370o63LHY1G4797KZD+9a9/9K9//aN//evfAGBi+LO0vPTnggsuSFdXl2rGmXPPPTfnn39+6/L81atX/x92R//61z/617/+0T/6NwAYx2688cbjkyxqXuvs7Mzv/u7vqmWc+tCHPpRp06btt1ZV1V+WZXmC3dG//vWP/tE/+kf/BgDj1NDQ0F+n5VM/r7zyynR2diplnJo6dWp+7/d+r3X5qCSL7Y7+9a9/9I/+0T/6NwAYh8qyvDDJHzevnX766TnvPJ8JM97NnTt3pE8FvaYsywvsjv7R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gHGmt7f3Z0n+oXntmWeeycqVKxUyzv385z/PunXrWpdv6e3t9eTr3+boH/2jf/SP/g0Axqnrk+xsXrjnnnt8Lcg4NjAwkB/+8Iety9uTLLI7+te//tE/+kf/6H+8a5uoP3hfX9+enp6evUmueHVtaGgoL774YmbPnq2Wceh73/tefvWrX+23VhTFtWVZPmB39K9//aN//esf/aP/8W4ivwIgVVX9bZL9vvZhxYoV6e/vV8s4s3r16qxatap1eXlVVf/V7uhf//pH//rXP/pH/wYA41xZlo1arfbvkww2r991113ZtWuXasaJXbt25c4772xdHqzVateUZdmwQ/rXv/7Rv/71j/71r/+JoG2iH477779/c09PT1uS7lfXhoeHs3nz5sydOzdFUShoDKuqKt/85jezdevW1rt6Fy1adIcd0r/+9Y/+9a9/9K9//U8UNUckqarqxrS8FOjpp59OX1+fzRnjfvSjH+W5555rXX6oq6vrZruD/vWP/vWvf/Svf/0bAEwwZVkOt7W1fTbJfq/7efDBB7N69WobNEY99dRTWbZsWevyzkmTJn12wYIFdTuE/vWP/vWvf/Svf/0bAExACxcufCbJ51vXv/vd72bLli02aIzZvHlz7rjjda/wqYqiuOZLX/rSc3YI/esf9K9/9K9//U80bbbgf+vr61vV3d19dJL3v7pWr9ezZs2azJkzJx0dHTZpDHjxxRfzta99baTvdP1PZVn+rR1C//oH/esf9K//icgrAF7vPyZ5sHlhx44due222zI0NGR3RrnBwcHceuut2blzZ+td93V1dV1vh9C//kH/+gf9698AgCQvvx8oySeTPN28vmHDhtx+++2p1711ZLQaHh7O7bffno0bN7betSbJp73vB/3rH/Svf9C//icybwEYQV9f357LLrvs7qqqPpNk2qvr27dvz6ZNm9LV1eXrQUaZRqORpUuXZu3ata13bUtyWVmW6+0S+tc/6F//oH/9GwDwOvfff//27u7u5Un+KMmk107Ttm3Ztm1bzj33XBeBURT/t7/97Tz55JOtd+2t1WpXLlq06HG7hP71D/rXP+hf/wYAvKG+vr7nuru7VyS5qnmvtmzZkueffz6zZ89OreZdFIdTvV7Pt771rZG+rmUoyVWLFi26zy6hf/2D/vUP+tc/BgBv5SLwVE9Pz9okH0/TZyZs27Yt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";

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
