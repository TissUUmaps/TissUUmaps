/**
 * @namespace markerUtils
 * @classdesc Work with anything that has to do with markers, take options from the interface
  about markers, and create markers  
   * @property {Array(String)}   markerUtils._d3Symbols -  
   * @property {Array(String)}   markerUtils._d3SymbolStrings - 
   * 
*/

markerUtils = {
    //type must be like d3.symbolVoss
    _d3Symbols: [d3.symbolCross, d3.symbolDiamond, d3.symbolSquare, d3.symbolTriangle, d3.symbolStar, d3.symbolWye, d3.symbolCircle],  // Not used
    _symbolStrings: ["cross", "diamond", "square", "triangle up", "star", "clobber", "disc", "hbar", "vbar", "tailed arrow", "triangle down", "ring", "x", "arrow", "gaussian"],
    _symbolUnicodes: ["＋ cross", "◆ diamond", "■ square", "▲ triangle up", "★ star", "✇ clobber", "● disc", "▬ hbar", "▮ vbar", "➔ tailed arrow", "▼ triangle down", "○ ring", "⨯ x", "> arrow", "◌ gaussian"],
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
markerUtils.makePiechartTable = function(uid, markerIndex, sectorsPropertyName) {
    const markerData = dataUtils.data[uid]["_processeddata"];

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
    if (dataUtils.data[uid]["_tooltip_fmt"]) {
        outText += "<b>" + markerUtils.getMarkerTooltip(uid, markerIndex) + ":</b><br/>";
    }
    let sectorValues = markerData[sectorsPropertyName][markerIndex].split(";");
    let sortedSectors = [];
    sectors.forEach(function (sector, index) {
        sortedSectors.push([parseFloat(sectorValues[index]), sector, index])
    });
    sortedSectors.sort(
        function cmp(a, b) {
            return b[0]-a[0];
        }
    );
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
