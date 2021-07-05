/**
* @file dataUtils.js Handling data for TissUUmaps
* @author Leslie Solorzano
* @see {@link dataUtils}
*/

/**
* @namespace dataUtils
* @property {Object} dataUtils._expectedCSV - Expected csv structure
* @property {Object} dataUtils._subsampledBarcodes - Containing the subsamples barcode trees to display
* @property {Array}  dataUtils._barcodesByAmount - Sorted list of arrays of the data
* @property {Number} dataUtils._maximumAmountInLowerRes - Maximum amount of points to display in low res
* @property {Object} dataUtils._nameAndLetters - Contains two bools drawGeneName and drawGeneLetters to know if and which to display between barcode or gene name
* @property {Object} dataUtils._drawOptions - Options for markerUtils.printBarcodeUIs */
dataUtils = {
    /** _CSVStructure - Expected csv structure */
    _CSVStructure: { headers: ["barcode", "gene_name", "global_X_pos", "global_Y_pos", "seq_quality_min"] },
    _expectedCSV: { "group": "macro_cluster", "name": "", "X_col": "global_X_pos", "Y_col": "global_Y_pos", "key": "" },
    _subsampledBarcodes: {},
    _barcodesByAmount: [],
    _maximumAmountInLowerRes: 5000,
    _nameAndLetters: { drawGeneName: false, drawGeneLetters: false },
    _drawOptions: { randomColorForMarker: false },
    _autoLoadCSV: false
    //_minimumAmountToDisplay: 500,
    //_subsamplingRate: 100,
}
/** 
* From the interface, get the key that will be used for nesting the raw data 
* and making my lovely quadtrees */
dataUtils.processISSRawData = function () {
    
    var imageWidth = OSDViewerUtils.getImageWidth();
    var op = tmapp["object_prefix"];

    var progressParent=interfaceUtils.getElementById("ISS_csv_progress_parent");
    if(progressParent == null){
        console.log("No progress bar present.")
    }else{
        progressParent.style.visibility="hidden";
        progressParent.style.display="none";
    }
    
    var ISSBarcodeInputNode = document.getElementById("ISS_barcode_header");
    var barcodeSelector = ISSBarcodeInputNode.options[ISSBarcodeInputNode.selectedIndex].value;
    var ISSNanmeInputNode = document.getElementById("ISS_name_header");
    var nameSelector = ISSNanmeInputNode.options[ISSNanmeInputNode.selectedIndex].value;
    var ISSXNode = document.getElementById("ISS_X_header");
    var xSelector = ISSXNode.options[ISSXNode.selectedIndex].value;
    var ISSYNode = document.getElementById("ISS_Y_header");
    var ySelector = ISSYNode.options[ISSYNode.selectedIndex].value;
    var ISSColor = document.getElementById("ISS_color_header");
    var ISSScale = document.getElementById("ISS_scale_header");
    var ISSPiechart = document.getElementById("ISS_piechart_header");
    if (ISSColor)
        var colorSelector = ISSColor.options[ISSColor.selectedIndex].value;
    else
        var colorSelector = "null";
    if (ISSScale)
        var scaleSelector = ISSPiechart.options[ISSScale.selectedIndex].value;
    else
        var scaleSelector = "null";
    if (ISSPiechart)
        var piechartSelector = ISSPiechart.options[ISSPiechart.selectedIndex].value;
    else
        var piechartSelector = "null";
    
    if (colorSelector && colorSelector != "null"){
        markerUtils._uniqueColor = true;
        markerUtils._uniqueColorSelector = colorSelector;
    }
    else {
        markerUtils._uniqueColor = false;
        markerUtils._uniqueColorSelector = "";
    }
    if (piechartSelector && piechartSelector != "null"){
        markerUtils._uniquePiechart = true;
        markerUtils._uniquePiechartSelector = piechartSelector;
    }
    else {
        markerUtils._uniquePiechart = false;
        markerUtils._uniquePiechartSelector = "";
    }
    if (scaleSelector && scaleSelector != "null"){
        markerUtils._uniqueScale = true;
        markerUtils._uniqueScaleSelector = scaleSelector;
    }
    else {
        markerUtils._uniqueScale = false;
        markerUtils._uniqueScaleSelector = "";
    }
    
    //check that the key is available
    var knode = document.getElementById(op + "_key_header");
    var key = knode.options[knode.selectedIndex].value;
    
    if (key.includes("letters")) {
        //make sure that the barcode column is selected
        if ((barcodeSelector == "null")) {
            //console.log("entered here");
            console.log("Key is selected to be Barcode but no column was selected in csv");
            if (!(nameSelector == "null")) {
                knode.options[1].selected = true;//option for gene name
                console.log("changing key to gene name");
            }
        }
    }
    
    if (key.includes("gene_name")) {
        //make sure that the barcode column is selected
        if ((nameSelector == "null")) {
            //console.log("entered here");
            console.log("Key is selected to be Gene Name but no column was selected in csv")
            if (!(barcodeSelector == "null")) {
                knode.options[0].selected = true; //option for barcode
                console.log("changing key to barcode");
            }
        }
    }
    
    //console.log("barcodeSelector nameSelector",barcodeSelector,nameSelector);
    
    if (!(nameSelector == "null")) {
        //console.log("entered here");
        dataUtils._nameAndLetters.drawGeneName = true;
    }
    else {
        dataUtils._nameAndLetters.drawGeneName = false;
    }
    if (!(barcodeSelector == "null")) {
        //console.log("entered here");
        dataUtils._nameAndLetters.drawGeneLetters = true;
    }
    else {
        //console.log("entered here");
        dataUtils._nameAndLetters.drawGeneLetters = false;
    }
    
    var toRemove = [barcodeSelector, nameSelector, xSelector, ySelector];
    var extraSelectors = []
    dataUtils._CSVStructure[op + "_csv_header"].forEach(function (item) { extraSelectors.push(item) });
    extraSelectors = extraSelectors.filter((el) => !toRemove.includes(el));
    dataUtils[op + "_processeddata"] = [];
    dataUtils[op + "_rawdata"].forEach(function (rawdatum) {
        var obj = {};
        obj["letters"] = rawdatum[barcodeSelector];
        obj["gene_name"] = rawdatum[nameSelector];
        obj["global_X_pos"] = Number(rawdatum[xSelector]);
        obj["global_Y_pos"] = Number(rawdatum[ySelector]);
        obj["viewer_X_pos"] = (obj["global_X_pos"] + 0.5) / imageWidth;
        obj["viewer_Y_pos"] = (obj["global_Y_pos"] + 0.5) / imageWidth;
        extraSelectors.forEach(function (extraSelector) {
            obj[extraSelector] = rawdatum[extraSelector];
        });
        dataUtils[op + "_processeddata"].push(obj);
    });
    
    dataUtils.makeQuadTrees();
    
    delete dataUtils[op + "_rawdata"];
    if (document.getElementById("ISS_globalmarkersize")) {
        document.getElementById("ISS_globalmarkersize").style.display = "block";
    }
    if (document.getElementById("ISS_searchmarkers_row")) {
        document.getElementById("ISS_searchmarkers_row").style.display = "block";
    }
    if (window.hasOwnProperty("glUtils")) {
        glUtils.loadMarkers();  // Update vertex buffers, etc. for WebGL drawing
    }
    if (markerUtils._uniquePiechartSelector != ""){
        markerUtils.addPiechartLegend();
    }
    else {
        document.getElementById("piechartLegend").style.display="none";
    }
}

/** 
* Set expected headers
*/
dataUtils.setExpectedCSV = function(expectedCSV){
    dataUtils._expectedCSV = expectedCSV;
}

/** 
* Show the menu do select the CSV headers that contain the information to display*/
dataUtils.showMenuCSV = function(){
    var op = tmapp["object_prefix"];
    var csvheaders = Object.keys(dataUtils[op + "_rawdata"][0]);
    dataUtils._CSVStructure[op + "_csv_header"] = csvheaders;
    var ISSBarcodeInput = document.getElementById(op + "_barcode_header");
    var ISSNanmeInput = document.getElementById(op + "_name_header");
    var ISSX = document.getElementById(op + "_X_header");
    var ISSY = document.getElementById(op + "_Y_header");
    var ISSColor = document.getElementById(op + "_color_header");
    var ISSScale = document.getElementById(op + "_scale_header");
    var ISSPiechart = document.getElementById(op + "_piechart_header");
    var ISSKey = document.getElementById(op + "_key_header");
    //console.log(dataUtils._CSVStructure["ISS_csv_header"]);
    [ISSBarcodeInput, ISSNanmeInput, ISSX, ISSY, ISSColor, ISSScale, ISSPiechart].forEach(function (node) {
        if (!node) return;
        node.innerHTML = "";
        var option = document.createElement("option");
        option.value = "null";
        option.text = "-----";
        node.appendChild(option);
        csvheaders.forEach(function (head) {
            var option = document.createElement("option");
            option.value = head;
            option.text = head.split(";")[0];
            node.appendChild(option);
        });
    });
    var panel = document.getElementById(op + "_csv_headers");
    if (!dataUtils._autoLoadCSV) {
        panel.style = "";
    }
    //search for defaults if any, "barcode" used to be called "letters"
    //it is still "letters in the obejct" but the BarcodeInputValue can be anything chosen by the user
    //and found in the csv column
    if (csvheaders.includes(dataUtils._expectedCSV["group"])) ISSBarcodeInput.value = dataUtils._expectedCSV["group"];
    if (csvheaders.includes(dataUtils._expectedCSV["name"])) ISSNanmeInput.value = dataUtils._expectedCSV["name"];
    if (csvheaders.includes(dataUtils._expectedCSV["X_col"])) ISSX.value = dataUtils._expectedCSV["X_col"];
    if (csvheaders.includes(dataUtils._expectedCSV["Y_col"])) ISSY.value = dataUtils._expectedCSV["Y_col"];
    if (csvheaders.includes(dataUtils._expectedCSV["color"])) ISSColor.value = dataUtils._expectedCSV["color"];
    if (csvheaders.includes(dataUtils._expectedCSV["piechart"])) ISSPiechart.value = dataUtils._expectedCSV["piechart"];
    if (csvheaders.includes(dataUtils._expectedCSV["scale"])) ISSScale.value = dataUtils._expectedCSV["scale"];
    if (dataUtils._expectedCSV["key"]) ISSKey.value = dataUtils._expectedCSV["key"];
    if (dataUtils._autoLoadCSV) {
        setTimeout(function () {
            document.getElementById(op + "_bringmarkers_btn").click();
        },500);
    }
}

/** 
* Creeate the dataUtils[op + "_barcodeGarden"] ("Garden" as opposed to "forest")
* To save all the trees per barcode or per key. It is an object so that it is easy to just call
* the right tree given the key instead of looping through an array. */
dataUtils.makeQuadTrees = function () {
    var op = tmapp["object_prefix"];
    var x = function (d) {
        return d["global_X_pos"];
    };
    var y = function (d) {
        return d["global_Y_pos"];
    };
    
    
    var op = tmapp["object_prefix"];
    var knode = document.getElementById(op + "_key_header");
    var key = knode.options[knode.selectedIndex].value;
    var allbarcodes = d3.nest().key(function (d) { return d[key]; }).entries(dataUtils[op + "_processeddata"]);
    console.log(allbarcodes);
    dataUtils[op + "_barcodeGarden"] = {};
    for (var i = 0; i < allbarcodes.length; i++) {
        var gardenKey = allbarcodes[i].key;
        var gene_name = allbarcodes[i].values[0].gene_name || "";
        var letters = allbarcodes[i].values[0].letters || "";
        //console.log(letters);
        dataUtils[op + "_barcodeGarden"][gardenKey] = d3.quadtree().x(x).y(y).addAll(allbarcodes[i].values);
        dataUtils[op + "_barcodeGarden"][gardenKey].treeName = letters;
        dataUtils[op + "_barcodeGarden"][gardenKey].treeGeneName = gene_name;
        //create the subsampled for all those that need it                  
    }
    dataUtils[op + "_data"] = [];
    allbarcodes.forEach(function (n) {
        dataUtils[op + "_data"].push(n);
    });
    markerUtils.printBarcodeUIs(dataUtils._drawOptions);
    var panel = document.getElementById(op+"_csv_headers");
    panel.style = "visibility: hidden; display:none;";
    
}

dataUtils.XHRCSV = function (thecsv) {
    var op = tmapp["object_prefix"];

    var panel = interfaceUtils.getElementById(op + "_csv_headers");
    panel.style.visibility="hidden"; 
    panel.style.display="none"

    var xhr = new XMLHttpRequest();

    var progressParent=interfaceUtils.getElementById("ISS_csv_progress_parent");
    progressParent.style.visibility="visible";
    progressParent.style.display="block";
    //console.log(progressParent)

    var progressBar=interfaceUtils.getElementById("ISS_csv_progress");
    var fakeProgress = 0;
    
    // Setup our listener to process compeleted requests
    xhr.onreadystatechange = function () {        
        // Only run if the request is complete
        if (xhr.readyState !== 4) return;        
        // Process our return data
        if (xhr.status >= 200 && xhr.status < 300) {
            // What do when the request is successful
            progressBar.style.width = "100%";
            dataUtils[op + "_rawdata"] = d3.csvParse(xhr.responseText);
            dataUtils.showMenuCSV();
            
        }else{
            console.log("dataUtils.XHRCSV responded with "+xhr.status);
            progressParent.style.display = "none";
            alert ("Impossible to load data, please contact an administrator.")
        }     
    };
    
    xhr.onprogress = function (pe) {
        if (pe.lengthComputable) {
            var maxsize = pe.total;
            var prog=pe.loaded;
            var perc=prog/maxsize*100;
            perc=perc.toString()+"%"
            progressBar.style.width = perc;
            //console.log(perc);
        }
        else {
            fakeProgress += 1;
            console.log(fakeProgress, Math.min(100, 100*(1-Math.exp(-fakeProgress/50.))))
            var perc=Math.min(100, 100*(1-Math.exp(-fakeProgress/50.)));
            perc=perc.toString()+"%"
            progressBar.style.width = perc;
        }
    }

    xhr.open('GET', thecsv);
    xhr.send();
    
}

/** 
* @param {File} thecsv 
* @return {Array} The csv headers.
* This reads the CSV and stores the raw data and sets the headers 
* in the interface and at [op + "_csv_header"]
* Later on it should be nested according to the main criteria */
dataUtils.readCSV = function (thecsv) {
    var op = tmapp["object_prefix"];
    var panel = interfaceUtils.getElementById(op + "_csv_headers");
    panel.style.visibility="hidden"; 
    panel.style.display="none"
    dataUtils[op + "_rawdata"] = {};
    dataUtils._CSVStructure[op + "_csv_header"] = null;
    var request = d3.csv(
        thecsv,
        function (d) { return d; },
        function (rows) {
            dataUtils[op + "_rawdata"] = rows;
            dataUtils.showMenuCSV();
        }
    );
}
    
/** 
* subsamples the full amount of barcodes so that in the lower resolutions only a significant portion
* is drawn and we don't wait and kill our browser.
* Sumsampling is done homogenously for all the space and density. 
* @param {Number} amount needed amount of barcodes
* @param {String} barcode Barcode or gene_name (key) to search for in op+_data*/
dataUtils.randomMarkersFromBarcode = function (amount, barcode) {
    var op = tmapp["object_prefix"];
    dataUtils[op + "_data"].forEach(function (bar) {
        if (bar.key == barcode) {
            var barcodes = bar.values;
            var maxindex = barcodes.length - 1;
            
            for (var i = 0; i <= maxindex; i++) {
                var index = Math.floor(Math.random() * (maxindex - i + 0)) + i;
                var temp = barcodes[i];
                barcodes[i] = barcodes[index];
                barcodes[index] = temp;
            }
            dataUtils._subsampledBarcodes[barcode] = barcodes.slice(0, amount);
        }
    });
}

/** 
* subsamples the full list from a barcode so that in the lower resolutions only a significant portion
* is drawn and we don't wait and kill our browser  
* @param {Number} amount needed amount of barcodes
* @param {barcodes[]} list A list */
dataUtils.randomSamplesFromList = function (amount, list) {
    //var op=tmapp["object_prefix"];
    if (amount >= list.length) return list;
    
    for (var i = 0; i < amount; i++) {
        var index = Math.floor(Math.random() * (list.length - i + 0 - 1)) + i;
        var temp = list[i];
        list[i] = list[index];
        list[index] = temp;
    }
    
    return list.slice(0, amount);
    
}

/** 
* Find all the markers for a specific key (name or barcode)  
* @param {string} keystring to search in op+_data usually letters like "AGGC" but can be the gene_name */
dataUtils.findBarcodesInRawData = function (keystring) {
    var op = tmapp["object_prefix"];
    var values = null;
    dataUtils[op + "_data"].forEach(function (input) {
        if (input.key == keystring) {
            values = input.values;
        }
    });
    return values;
}

/** 
* Take dataUtils[op + "_data"] and sort it (permanently). 
* Calculate and save the right amount of downsampling for each barcode.
* And save the subsample arrays, subsampling is homogeneous   */
dataUtils.sortDataAndDownsample = function () {
    var op = tmapp["object_prefix"];
    var compareKeys = function (a, b) {
        if (a.values.length > b.values.length) { return -1; }
        if (a.values.length < b.values.length) { return 1; }
        return 0;
    };
    
    dataUtils[op + "_data"].sort(compareKeys);
    
    //take the last element of the list which has the minimum amount
    var minamount = dataUtils[op + "_data"][dataUtils[op + "_data"].length - 1].values.length;
    //take the first element of the list which has the maximum amount
    var maxamount = dataUtils[op + "_data"][0].values.length;
    //total amount of barcodes
    var amountofbarcodes = dataUtils[op + "_data"].length;
    
    dataUtils[op + "_data"].forEach(function (barcode) {
        var normalized = (barcode.values.length - minamount) / maxamount;
        var downsize = dataUtils._maximumAmountInLowerRes * Math.log(barcode.values.length) / Math.log(maxamount);
        if (downsize > barcode.values.length) { downsize = barcode.values.length; }
        dataUtils._barcodesByAmount.push({ "barcode": barcode.key, "amount": barcode.values.length, "normalized": normalized, "downsize": downsize });
    });
    
    dataUtils._barcodesByAmount.forEach(function (b) {
        dataUtils.randomMarkersFromBarcode(b.downsize, b.barcode);
    });
}