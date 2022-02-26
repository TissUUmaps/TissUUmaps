/**
 * @namespace markerUtils
 * @classdesc Work with anything that has to do with markers, take options from the interface
  about markers, and create markers  
   * @property {Bool}   markerUtils._drawPaths -draw D3 symbols (true)  or a D3 rect (false)
   * @property {Number} markerUtils._globalMarkerSize - 
   * @property {Number} markerUtils._uniqueColor - Keep then number of drawn regions and also let them be the id, 
   * @property {String} markerUtils._uniqueColorSelector - 
   * @property {Bool}   markerUtils._uniqueScale -
   * @property {String} markerUtils._uniqueScaleSelector -
   * @property {Bool}   markerUtils._uniquePiechart -
   * @property {String} markerUtils._uniquePiechartSelector -
   * @property {Number} markerUtils._startCullingAt - 
   * @property {Obj}    markerUtils._checkBoxes - 
   * @property {Array(String)}   markerUtils._d3Symbols -  
   * @property {Array(String)}   markerUtils._d3SymbolStrings - 
   * * @property {Object}  markerUtils._colorsperkey - load colors per key if known previously 
   * 
*/

markerUtils = {
    //type must be like d3.symbolVoss
    _drawPaths: true,
    _globalMarkerSize: 1,
    _showSizeColumn: false,
    _uniqueColor:false, //if this and selector are true, it will try to find a color unique to each spot
    _uniqueColorSelector:null, //is a string of the type "[float,float,float]" that gets converted to a string "rgb(uint8,uint8,uint8)"
    _uniqueScale:false, //if this and selector are true, it will try to find a color unique to each spot
    _uniqueScaleSelector:null, //is a string of the type "[float,float,float]" that gets converted to a string "rgb(uint8,uint8,uint8)"
    _uniquePiechart:false, //if this and selector are true, it will try to show a unique piechart for each spot
    _uniquePiechartSelector:null, //a string with the name of the piechart data field in the CSV
    _startCullingAt: 9000,
    _checkBoxes: {},
    _d3Symbols: [d3.symbolCross, d3.symbolDiamond, d3.symbolSquare, d3.symbolTriangle, d3.symbolStar, d3.symbolWye, d3.symbolCircle],  // Not used
    _symbolStrings: ["cross", "diamond", "square", "triangle up", "star", "clobber", "disc", "hbar", "vbar", "tailed arrow", "triangle down", "ring", "x", "arrow"],
    _symbolUnicodes: ["＋ cross", "◆ diamond", "■ square", "▲ triangle up", "★ star", "✇ clobber", "● disc", "▬ hbar", "▮ vbar", "➔ tailed arrow", "▼ triangle down", "○ ring", "⨯ x", "> arrow"],
    _colorsperkey:null,
    _startMarkersOn:false,
    _randomShape:true,
    _selectedShape:0,
    _headerNames:{"Barcode":"Barcode","Gene":"Gene"}
}

/** In the markers interface, hide all the rows that do not contain the search string 
 *  specified in the interface in the textarea
*/
markerUtils.hideRowsThatDontContain = function () {
    var op = tmapp["object_prefix"];
    var contains = function (row, searchFor) {
        var v = row.textContent.toLowerCase();
        var v2 = searchFor;
        if (v2) {
            v2 = v2.toLowerCase();
        }
        return v.indexOf(v2) > -1;
    };

    var aneedle = document.getElementById(op + "_search").value;
    var rows = document.getElementById(op + "_table").rows;

    //make it so that the needle can be a list separated by comma, no spaces

    console.log(aneedle);

    var needles=[];

    if (aneedle.indexOf(',') > -1) { 
        needles=aneedle.split(',');
    }else{
        needles.push(aneedle)
    }

    
    for (var i = 2; i < rows.length; i++) {
        var show=false;
        needles.forEach(function(needle){
            if (contains(rows[i], needle)) {
                show=true;
            }
        });
        if (!show) {
            rows[i].setAttribute("style", "display:none;");
        } else { rows[i].setAttribute("style", ""); }
    }
}

/** Show all rows from the markers UI again */
markerUtils.showAllRows = function () {
    var op = tmapp["object_prefix"];
    var rows = document.getElementById(op + "_table").rows;

    for (var i = 0; i < rows.length; i++) {
        rows[i].setAttribute("style", "");
    }
}

/** Adding piechart legend in the upper left corner */
markerUtils.updatePiechartLegend = function() {

    if (document.getElementById("piechartLegend") == undefined) {
        let elt = document.createElement("div");
        elt.className = "piechartLegend px-1 mx-1 viewer-layer"
        elt.id = "piechartLegend"
        tmapp['ISS_viewer'].addControl(elt,{anchor: OpenSeadragon.ControlAnchor.TOP_LEFT});
    }
    elt = document.getElementById("piechartLegend");
    elt.style.display="block";
    elt.innerHTML = "";

    let markerData = undefined, sectorsPropertyName = undefined;
    for (let [uid, numPoints] of Object.entries(glUtils._numPoints)) {
        if (glUtils._usePiechartFromMarker[uid]) {
            markerData = dataUtils.data[uid]["_processeddata"];
            sectorsPropertyName = dataUtils.data[uid]["_pie_col"];
        }
    }
    if (markerData == undefined || sectorsPropertyName == undefined) return;

    let table = HTMLElementUtils.createElement({ kind: "table" });
    table.style.borderSpacing = "3px";
    table.style.borderCollapse = "separate";
    table.style.fontSize = "10px";
    let title = HTMLElementUtils.createElement({ kind: "div", innerHTML: "<b>Piechart legend</b>"});
    elt.appendChild(title);
    elt.appendChild(table);

    let sectors = [];
    const numSectors = markerData[sectorsPropertyName][0].split(";").length;
    if (sectorsPropertyName.split(";").length == numSectors) {
        sectors = sectorsPropertyName.split(";");  // Use sector labels from CSV header
    } else {
        for (let i = 0; i < numSectors; i++) {
            sectors.push("Sector " + (i+1));  // Assign default sector labels
        }
    }

    sectors.forEach(function (sector, index) {
        let row = HTMLElementUtils.createElement({ kind: "tr"});
        row.style.paddingBottom = "4px";
        let colortd = HTMLElementUtils.createElement({ kind: "td", innerHTML: "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"});
        colortd.style.backgroundColor = glUtils._piechartPalette[index % glUtils._piechartPalette.length];
        colortd.style.maxWidth = "70px";
        colortd.style.borderWidth= "1px";
        colortd.style.borderColor= "black";
        colortd.style.borderStyle= "solid";
        let labeltd = HTMLElementUtils.createElement({ kind: "td", innerHTML: "&nbsp;" + sector});
        row.appendChild(colortd);
        row.appendChild(labeltd);
        table.appendChild(row);
    })
    console.log(sectors);
}

/** Adding piechart table on pickup */
markerUtils.makePiechartTable = function(markerData, markerIndex, sectorsPropertyName) {

    let sectors = [];
    const numSectors = markerData[sectorsPropertyName][markerIndex].split(";").length;
    if (sectorsPropertyName.split(";").length == numSectors) {
        sectors = sectorsPropertyName.split(";");  // Use sector labels from CSV header
    } else {
        for (let i = 0; i < numSectors; i++) {
            sectors.push("Sector " + (i+1));  // Assign default sector labels
        }
    }

    let outText = "";
    let sectorValues = markerData[sectorsPropertyName][markerIndex].split(";");
    let sortedSectors = [];
    sectors.forEach(function (sector, index) {
        sortedSectors.push([parseFloat(sectorValues[index]), sector, index])
    });
    console.dir(sortedSectors);
    sortedSectors.sort(
        function cmp(a, b) {
            return b[0]-a[0];
        }
    );
    console.dir(sortedSectors);
    sortedSectors.forEach(function (sector) {
        outText += "<span style='border:2px solid " + glUtils._piechartPalette[sector[2] % glUtils._piechartPalette.length] + ";padding:3px;margin:2px;display: inline-block;'>" + sector[1] + ": " + (sector[0] * 100).toFixed(1) + " %</span> ";
    })
    return outText;
}

/** Get tooltip format on pickup */
markerUtils.getMarkerTooltip = function(uid, markerIndex) {
    // "{tab}","{key}","{name}","{index}","{color}", "{col_...}"
    const formatString = dataUtils.data[uid]["_tooltip_fmt"];
    const tabName = interfaceUtils.getElementById(uid + "_marker-tab-name").textContent;
    const markerData = dataUtils.data[uid]["_processeddata"];
    const keyName = dataUtils.data[uid]["_gb_col"];
    const keyVal = (keyName != null) ? markerData[keyName][markerIndex] : undefined;
    const nameName = dataUtils.data[uid]["_gb_name"];
    const nameVal = (nameName != null) ? markerData[nameName][markerIndex] : undefined;
    const colName = dataUtils.data[uid]["_cb_col"];
    const colVal = (colName != null) ? markerData[colName][markerIndex] : undefined;
    
    var returnString = formatString;
    if (returnString == "") {
        if (keyVal !== undefined) {
            return keyVal;
        }
        else if (colName !== undefined) {
            return colVal;
        }
    }
    returnString=returnString.replace('{index}', markerIndex);
    returnString=returnString.replace('{tab}', tabName);
    returnString=returnString.replace('{key}', keyVal);
    returnString=returnString.replace('{name}', nameVal);
    returnString=returnString.replace('{color}', colVal);
    for (var header of dataUtils.data[uid]["_csv_header"]) {
        var headerVal = markerData[header][markerIndex];
        returnString=returnString.replace('{col_'+header+'}', headerVal);
    }
    return returnString;
}
