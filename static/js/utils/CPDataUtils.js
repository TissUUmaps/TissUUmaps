/**
 * @namespace CPDataUtils
 * @classdesc Sets of functions to prepare and preload data information from Cell profiler
*/
CPDataUtils = {
    //                      String: a key, X , Y, Q
    _CSVStructure: { headers: ["letters", "gene_name", "global_X_pos", "global_Y_pos", "seq_quality_min"] },
    _d3LUTs:["ownColorFromColumn","interpolateCubehelixDefault", "interpolateRainbow", "interpolateWarm", "interpolateCool", "interpolateViridis", "interpolateMagma", "interpolateInferno", "interpolatePlasma", "interpolateRdYlGn", "interpolateBuGn", "interpolateBuPu", "interpolateGnBu", "interpolateOrRd", "interpolatePuBuGn", "interpolatePuBu", "interpolatePuRd", "interpolateRdPu", "interpolateYlGnBu", "interpolateYlGn", "interpolateYlOrBr", "interpolateYlOrRd", "interpolateBlues", "interpolateGreens", "interpolateGreys", "interpolatePurples", "interpolateReds", "interpolateOranges"],
    _subsampledItems: {},
    _ownColorLut:{"class":"hexcolor"},
    _barcodesByAmount: [],
    _subsamplingRate: 100,
    _minimumAmountToDisplay: 500,
    _markersize:0.0008,
    _subsamplingfactor:0.15,
    _drawCPdata: false,
    _expectedCSV: { "key_header": "Global_Exp_ID", "X_header": "Global_X", "Y_header": "Global_Y", "colorscale": "interpolateRainbow" },
}

/** 
 * From the interface, get the key that will be used for nesting the raw data 
 * and making the quadtrees*/
CPDataUtils.processISSRawData = function () {
    
    var cpop="CP";
    var progressParent=interfaceUtils.getElementById("ISS_CP_csv_progress_parent");
    if(progressParent == null){
        console.log("No progress bar present.")
    }else{
        progressParent.style.visibility="hidden";
        progressParent.style.display="none";
    }
    var CPProperty = document.getElementById(cpop+"_property_header");
    var propertyselector=CPProperty.value;
    var CPX = document.getElementById(cpop+"_X_header");
    var xselector=CPX.value;
    var CPY = document.getElementById(cpop+"_Y_header");
    var yselector=CPY.value;
    var CPLut = document.getElementById(cpop+"_colorscale");
    var interpFunction=CPLut.value;
    
    var x = function (d) {
        return d[xselector];
    };
    var y = function (d) {
        return d[yselector];
    };
    if(!CPDataUtils[cpop + "_tree"])
        CPDataUtils[cpop + "_tree"] = d3.quadtree().x(x).y(y).addAll(CPDataUtils[cpop + "_rawdata"]);  
    CPDataUtils._drawCPdata=!tmapp["hideSVGMarkers"];  // SVG markers should not be drawn when WebGL is used
    markerUtils.drawCPdata({searchInTree:false}); //mandatory options obj
    
    if (document.getElementById("ISS_globalmarkersize")) {
        document.getElementById("ISS_globalmarkersize").style.display = "block";
    }
    if (window.hasOwnProperty("glUtils")) {
        glUtils.loadCPMarkers();  // Update vertex buffers, etc. for WebGL drawing
    }
}

/** 
* Set expected headers
*/
CPDataUtils.setExpectedCSV = function(expectedCSV){
    CPDataUtils._expectedCSV = expectedCSV;
}

CPDataUtils.readCSV = function (thecsv) {
    var cpop = "CP";//tmapp["object_prefix"];
    CPDataUtils[cpop + "_rawdata"] = {};
    CPDataUtils[cpop + "_rawdata_stats"]={};
    CPDataUtils._CSVStructure[cpop + "_csv_header"] = null;

    var progressParent=interfaceUtils.getElementById("ISS_CP_csv_progress_parent");
    progressParent.style.visibility="visible";
    progressParent.style.display="block";
    //console.log(progressParent)
    var progressBar=interfaceUtils.getElementById("ISS_CP_csv_progress");
    var fakeProgress = 0;

    var request = d3.csv(
        thecsv,
        function (d) { return d; },
        function (rows) {
            progressBar.style.width = "100%";
            CPDataUtils[cpop + "_rawdata"] = rows;
            CPDataUtils.loadFromRawData();
        }
    ).on("progress", function(pe){
        //update progress bar
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
    });;
}

CPDataUtils.loadFromRawData = function () {
    var cpop = "CP";//tmapp["object_prefix"];
    CPDataUtils[cpop + "_rawdata_stats"]={};
    var csvheaders = Object.keys(CPDataUtils[cpop + "_rawdata"][0]);
    CPDataUtils._CSVStructure=csvheaders;

    var datum=CPDataUtils[cpop + "_rawdata"][1];

    var numericalheaders=[];

    var rg=RegExp('^[0-9]*[.0-9]*$');
    //Check which headers could require stats:
    csvheaders.forEach(function(h){
        if(rg.test(datum[h])){
            //if it is not nan it means it is a number...
            numericalheaders.push(h);
            CPDataUtils[cpop + "_rawdata_stats"][h]={"min":+Infinity,"max":-Infinity,"mean":0}; 
        }
    });

    CPDataUtils[cpop + "_rawdata"].forEach(function(d){
        numericalheaders.forEach(function(nh){
            if(d[nh]>CPDataUtils[cpop + "_rawdata_stats"][nh]["max"]) CPDataUtils[cpop + "_rawdata_stats"][nh]["max"]=d[nh];
            if(d[nh]<CPDataUtils[cpop + "_rawdata_stats"][nh]["min"]) CPDataUtils[cpop + "_rawdata_stats"][nh]["min"]=d[nh];                    
        }); 
    });

    var CPKey = document.getElementById(cpop+"_key_header");
    var CPProperty = document.getElementById(cpop+"_property_header");
    var CPX = document.getElementById(cpop+"_X_header");
    var CPY = document.getElementById(cpop+"_Y_header");
    var CPLut = document.getElementById(cpop+"_colorscale");

    CPDataUtils._d3LUTs.forEach(function(lut){
        var option = document.createElement("option");
        option.value = lut;
        option.text = lut.replace("interpolate","");
        CPLut.appendChild(option);
    });
        

    [CPKey, CPProperty, CPX, CPY].forEach(function (node) {
        node.innerHTML = "";
        var option = document.createElement("option");
        option.value = "null";
        option.text = "-----";
        node.appendChild(option);
        csvheaders.forEach(function (head) {
            var option = document.createElement("option");
            option.value = head;
            option.text = head;
            node.appendChild(option);
        });
    });
    var panel = document.getElementById(cpop+"_csv_headers");
    panel.style = "";

    //create tree, full and subsampled array
    var length=CPDataUtils[cpop + "_rawdata"].length;
    var amount=Math.floor(length*CPDataUtils._subsamplingfactor);
    CPDataUtils[cpop + "_subsampled_data"]=CPDataUtils.randomSamplesFromList(amount,CPDataUtils[cpop + "_rawdata"]);

    //create listener for change of property
    var changeProperty=function(){
        //find all CP nodes and remove them
        for(d3nodename in overlayUtils._d3nodes){
            if(d3nodename.includes(cpop+"_prop_")){
                overlayUtils._d3nodes[d3nodename].selectAll("*").remove();
            }
        }
        markerUtils.drawCPdata({searchInTree:false});
    }
    CPProperty.addEventListener("change", changeProperty);
    
    if (csvheaders.includes(CPDataUtils._expectedCSV["key_header"])) CPKey.value = CPDataUtils._expectedCSV["key_header"];
    if (csvheaders.includes(CPDataUtils._expectedCSV["property_header"])) CPProperty.value = CPDataUtils._expectedCSV["property_header"];
    if (csvheaders.includes(CPDataUtils._expectedCSV["X_header"])) CPX.value = CPDataUtils._expectedCSV["X_header"];
    if (csvheaders.includes(CPDataUtils._expectedCSV["Y_header"])) CPY.value = CPDataUtils._expectedCSV["Y_header"];
    CPLut.value = "interpolateRainbow";
}

/** 
 * Remove the cp data from the view
 */
CPDataUtils.removeCPdata=function(){
    var cpop = "CP";//tmapp["object_prefix"];
    for(d3nodename in overlayUtils._d3nodes){
        if(d3nodename.includes(cpop+"_prop_")){
            overlayUtils._d3nodes[d3nodename].selectAll("*").remove();
        }
    }
}

/** 
 * subsamples the full list from a barcode so that in the lower resolutions only a significant portion
 * is drawn and we don't wait and kill our browser  
 * @param {Number} amount needed amount of barcodes
 * @param {data[]} list A list */
CPDataUtils.randomSamplesFromList = function (amount, list) {
    //var cpop=tmapp["object_prefix"];
    if(amount >= list.length) return list;

    for(var i=0;i<amount;i++){
      var index=Math.floor(Math.random() * (list.length - i + 0-1)) +i;
      var temp=list[i];
      list[i]=list[index];
      list[index]=temp;
    }
   
    return list.slice(0,amount);
}

/** 
* Find all CP elements in a box  */
CPDataUtils.arrayOfElementsInBox = function (x0, y0, x3, y3, options) {
    var cpop = "CP";

    var xselector = options.xselector;
    var yselector = options.yselector;
    
    var pointsInside = [];
    
    CPDataUtils[cpop + "_tree"].visit(function (node, x1, y1, x2, y2) {
        if (!node.length) {
            do {
                var d = node.data;
                d.scanned = true;
                var selected = (d[xselector] >= x0) && (d[xselector] < x3) && (d[yselector] >= y0) && (d[yselector] < y3);
                if (selected) {
                    pointsInside.push(d);
                }
            } while (node = node.next);
        }
        return x1 >= x3 || y1 >= y3 || x2 < x0 || y2 < y0;
    });
    return pointsInside;
}
