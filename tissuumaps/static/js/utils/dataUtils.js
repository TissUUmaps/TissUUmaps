/**
* @file dataUtils.js Handling data for TissUUmaps
* @author Leslie Solorzano
* @see {@link dataUtils}
*/

/**
 * @namespace dataUtils
 * @property {Object} dataUtils.data contains all data per tab
 * @property {array} dataUtils._d3LUTs All options ofr colormaps coming from d3
 */
dataUtils = {
    data:{
        /*"gene":{
            _type: "GENE_DATA"
        },
        "morphology":{
            _type: "MORPHOLOGY_DATA",
            _name:""
        },*/
        /*
        "U23423R":{
            _type: "GENERIC_DATA",
            _name:"",
            _processeddata:[],
            //iff user selects by group
            _groupgarden:[]// full of separated d3.tree
            _X:
            _Y:
            _gb_sr:
            _gb_col:
            _gb_name:
            _cb_cmap:
            _cb_col:
            _selectedOptions:{}
        }
        data_id:{kv pairs}
        ... and inifinitely more data "types" like piecharts or whatever
        */
    },
    _d3LUTs:[ "interpolateCubehelixDefault", "interpolateRainbow", "interpolateWarm", "interpolateCool", "interpolateViridis", 
    "interpolateMagma", "interpolateInferno", "interpolatePlasma", "interpolateBlues", "interpolateBrBG", "interpolateBuGn", "interpolateBuPu", "interpolateCividis", 
    "interpolateGnBu", "interpolateGreens", "interpolateGreys", "interpolateOrRd", "interpolateOranges", "interpolatePRGn", "interpolatePiYG", "interpolatePuBu", 
    "interpolatePuBuGn", "interpolatePuOr", "interpolatePuRd", "interpolatePurples", "interpolateRdBu", "interpolateRdGy", "interpolateRdPu", "interpolateRdYlBu", 
    "interpolateRdYlGn", "interpolateReds", "interpolateSinebow", "interpolateSpectral", "interpolateTurbo", "interpolateYlGn", "interpolateYlGnBu", "interpolateYlOrBr", 
    "interpolateYlOrRd"],
}
/**
 * BIG CHANGE
 * TODO: implemente createDataset so that Fredrik can draw
 * load data as usual, object format
 * trees only when user selects it, keep processeddata
 * how to close 
 * 
 * deterministic shapes
 * 
*/

/** 
* @param {String} uid The id of the data group
* @param {Object} options options that could be used inside, only holds name currently
* Creates an object inside dataUtils.data so that all options can be grouped by csv */
dataUtils.createDataset = function(uid,options){
    if(!options) options={};
    dataUtils.data[uid]={
        _type: "GENERIC_DATA",
        _name:options.name || "",
        _processeddata:[],
        //iff user selects by group
        _groupgarden:{},// full of separated d3.tree
        _X:"",
        _Y:"",
        _gb_sr:"",
        _gb_col:"",
        _gb_name:"",
        _cb_cmap:"",
        _cb_col:""
    }
}

/** 
* @param {HTMLEvent} event the event arries the file picker. which MUST have the data id in the begining separated by an "_"
* Selects a data object and reads a csv and starts the whole process to add it in datautils and the interface. 
* It can only be associated to a filepicker. It is atomatically listened when created in the interfaceUtils in the change event
*/
dataUtils.startCSVcascade= function(event){
    var data_id=event.target.id.split("_")[0];
    var file = event.target.files[0];
    if (file) {
        var reader = new FileReader();
        reader.onloadend = function (evt) {
            var dataUrl = evt.target.result;
            dataUtils.readCSV(data_id,dataUrl);
        };
        reader.readAsDataURL(file);
    }
}

/**
 * @deprecated Not required anymore, but kept for backwards-compatibility
 */
CPDataUtils={};

/** 
* @param {String} data_id The id of the data group like "U234345"
* @param {Array} data data coming from d3 after parsing the csv
* created the _processeddata list to be used in rendering
*/
dataUtils.processRawData = function(data_id,data) {
    let data_obj = dataUtils.data[data_id];

    data_obj["_processeddata"]=data;

    //this function is in case we need to standardize the data column names somehow,
    //so that the processseddata has some desired structure, but for now maybe no

    dataUtils.createMenuFromCSV(data_id,data[0]);

}

/** 
* @param {String} data_id The id of the data group like "U234345"
* Make sure that the options selected are correct an call the necessary functions to process the data so
* its ready to be displayed.
*/
dataUtils.updateViewOptions = function(data_id){

    var data_obj = dataUtils.data[data_id];
    
    if(data_obj === undefined){
        message="Load data first";
        interfaceUtils.alert(message); console.log(message);
        return;
    }

    var _selectedOptions = interfaceUtils._mGenUIFuncs.areRadiosAndChecksChecked(data_id);
    data_obj["_selectedOptions"]=_selectedOptions
    
    var radios = interfaceUtils._mGenUIFuncs.getTabRadiosAndChecks(data_id);
    var inputs = interfaceUtils._mGenUIFuncs.getTabDropDowns(data_id);

    //console.log(_selectedOptions);
    //console.log(radios);
    //console.log(inputs);

    if(inputs["X"].value == 'null' || inputs["Y"].value == 'null'){
        message="Select X and Y first";
        interfaceUtils.alert(message); console.log(message);
        return;
    }else{
        data_obj["_X"]=inputs["X"].value;
        data_obj["_Y"]=inputs["Y"].value;
    }
    console.log(tmapp["ISS_viewer"].world._items.length == 0, tmapp["ISS_viewer"].world._items);
    if (tmapp["ISS_viewer"].world._items.length == 0) {
        function getMax(arr) {
            let len = arr.length; let max = -Infinity;
            while (len--) { max = arr[len] > max ? arr[len] : max; }
            return max;
        }
        function getMin(arr) {
            let len = arr.length; let min = Infinity;
            
            while (len--) { min = arr[len] < min ? arr[len] : min; }
            return min;
        }
        minX = getMin(dataUtils.data[data_id]["_processeddata"].map(function(o) { return parseFloat(o[data_obj["_X"]]); }));
        maxX = getMax(dataUtils.data[data_id]["_processeddata"].map(function(o) { return parseFloat(o[data_obj["_X"]]); }));
        minY = getMin(dataUtils.data[data_id]["_processeddata"].map(function(o) { return parseFloat(o[data_obj["_Y"]]); }));
        maxY = getMax(dataUtils.data[data_id]["_processeddata"].map(function(o) { return parseFloat(o[data_obj["_Y"]]); }));
        console.log(minX,maxX, minY,maxY);
        if (minX <0 || maxX < 500) {
            for (o of dataUtils.data[data_id]["_processeddata"]) {
                o[data_obj["_X"]] = 1200 * (o[data_obj["_X"]] - minX) / (maxX - minX);
            }
            maxX = getMax(dataUtils.data[data_id]["_processeddata"].map(function(o) { return o[data_obj["_X"]]; }))
            console.log("new maxX,",maxX);
        }
        if (minY <0 || maxY < 500) {
            for (o of dataUtils.data[data_id]["_processeddata"]) {
                o[data_obj["_Y"]] = 1200 * (o[data_obj["_Y"]] - minY) / (maxY - minY);
            }
            maxY = getMax(dataUtils.data[data_id]["_processeddata"].map(function(o) { return o[data_obj["_Y"]]; }))
            console.log("new maxY,",maxY);
        }
        // We load an empty image at the size of the data.

        tmapp["ISS_viewer"].addTiledImage ({
            tileSource: {
                getTileUrl: function(z, x, y){return null},
                height: parseInt(maxY*1.06),
                width:  parseInt(maxX*1.06),
                tileSize: 256,
            },
            opacity: 0,
            x: -0.02,
            y: -0.02
        })
        setTimeout (function() {dataUtils.updateViewOptions(data_id)},50);
        return;
    }
    //this will be trickier since trees need to be made and also a menu
    
    if(inputs["gb_col"].value && inputs["gb_col"].value != "null"){
        data_obj["_gb_col"]=inputs["gb_col"].value;    
    }else{
        data_obj["_gb_col"]=null;    
    }

    if(inputs["gb_name"].value && inputs["gb_name"].value != "null"){
        data_obj["_gb_name"]=inputs["gb_name"].value;    
    }else{
        data_obj["_gb_name"]=null;    
    }

    // Load all settings inside of data_obj for easy access from glUtils
    // adds: data_obj["_cb_col"], data_obj["_cb_cmap"]
    //       data_obj["_pie_col"], data_obj["_scale_col"], data_obj["_shape_col"]
    if (radios["cb_gr"].checked) { // Color by group
        data_obj["_cb_col"]=null;
        data_obj["_cb_cmap"]=null;
    }
    else if (radios["cb_col"].checked) { // Color by marker
        if (inputs["cb_col"].value != "null") {
            if (inputs["cb_cmap"].value != "") {
                data_obj["_cb_cmap"]=inputs["cb_cmap"].value;
            }
            else {
                data_obj["_cb_cmap"]=null;
            }
            data_obj["_cb_col"]=inputs["cb_col"].value;
        }
        else  {
            interfaceUtils.alert("No color column selected. Impossible to update view.");return;
        }
    }
    // Use piecharts column
    data_obj["_pie_col"]=(radios["pie_check"].checked ? inputs["pie_col"].value : null);
    if (data_obj["_pie_col"]=="null") {
        interfaceUtils.alert("No piechart column selected. Impossible to update view.");return;
    }
    // Use scale colummn
    data_obj["_scale_col"]=(radios["scale_check"].checked ? inputs["scale_col"].value : null);
    if (data_obj["_scale_col"]=="null") {
        interfaceUtils.alert("No size column selected. Impossible to update view.");return;
    }
    data_obj["_scale_factor"]=inputs["scale_factor"].value;
    // Use shape column
    data_obj["_shape_col"]=(radios["shape_col"].checked ? inputs["shape_col"].value : null);
    if (data_obj["_shape_col"]=="null") {
        interfaceUtils.alert("No shape column selected. Impossible to update view.");return;
    }
    // Marker opacity
    data_obj["_opacity"]=inputs["opacity"].value;
    
    //this function veryfies if a tree with these features exist and doesnt recreate it
    dataUtils.makeQuadTrees(data_id);
    //print a menu in the interface for the groups
    table=interfaceUtils._mGenUIFuncs.groupUI(data_id);
    console.log(table);
    menuui=interfaceUtils.getElementById(data_id+"_menu-UI");
    menuui.classList.remove("d-none")
    menuui.innerText="";

    menuui.appendChild(table);
    sorttable.makeSortable(table);
    if(data_obj["_gb_col"]){
        var myTH = table.getElementsByTagName("th")[1];
        sorttable.innerSortFunction.apply(myTH, []);
    }
    //shape UXXXX_grname_shape, color UXXXX_grname_color

    // Make sure that slider for global marker size is shown
    if (interfaceUtils.getElementById("ISS_globalmarkersize"))
        interfaceUtils.getElementById("ISS_globalmarkersize").classList.remove("d-none");

    glUtils.loadMarkers(data_id);
    glUtils.draw();
}

/** 
* @param {String} data_id The id of the data group like "U234345"
* @param {Object} datumExample example datum that contains the headers of the csv
* Fills the necessary input dropdowns with the csv headers so that user can choose them
*/
dataUtils.createMenuFromCSV = function(data_id,datumExample) {
    var data_obj = dataUtils.data[data_id];

    var csvheaders = Object.keys(datumExample);
    data_obj["_csv_header"] = csvheaders;

    //fill dropdowns
    var alldrops=interfaceUtils._mGenUIFuncs.getTabDropDowns(data_id);
    var namesymbols=Object.getOwnPropertyNames(alldrops);
    namesymbols.forEach((drop)=>{
        if(drop=="cb_cmap") return; //if its colormaps dont fill it with csv but with d3 luts which are already there
        if(drop=="shape_fixed") return; //if its shapes dont fill it with csv but with shape symbols which are already there
        if(drop=="scale_factor" || drop=="shape_gr_dict" || drop=="cb_gr_dict" || drop=="opacity") return; //not dropdowns
        if (!alldrops[drop]) return;
        alldrops[drop].innerHTML = "";
        var option = document.createElement("option");
        option.value = "null"; option.text = "-----";
        alldrops[drop].appendChild(option);
        csvheaders.forEach(function (head) {
            var option = document.createElement("option");
            option.value = head; option.text = head.split(";")[0];
            alldrops[drop].appendChild(option);
        });
    })
    if (data_obj["expectedHeader"]) {
        interfaceUtils._mGenUIFuncs.fillRadiosAndChecksIfExpectedCSV(data_id,data_obj["expectedRadios"]);
        interfaceUtils._mGenUIFuncs.fillDropDownsIfExpectedCSV(data_id,data_obj["expectedHeader"]);
        dataUtils.updateViewOptions(data_id);
    }
}

/** 
* @param {String} data_id The id of the data group like "U234345"
* @param {Object} thecsv csv file path
* Calls dataUtils.createDataset and loads and parses the csv using D3. 
* then calls dataUtils.createMenuFromCSV to modify the interface in its own tab
*/
dataUtils.readCSV = function(data_id, thecsv) {
    
    dataUtils.createDataset(data_id,{"name":data_id});

    let data_obj = dataUtils.data[data_id];

    data_obj["_rawdata"] = {};
    data_obj["_csv_header"] = null;
    data_obj["_csv_path"] = thecsv;

    var progressParent=interfaceUtils.getElementById(data_id+"_csv_progress_parent");
    progressParent.classList.remove("d-none");
    var progressBar=interfaceUtils.getElementById(data_id+"_csv_progress");

    var fakeProgress = 0;

    var request = d3.csv(
        thecsv,
        function (d) { return d; } //here you can modify the datum 
    ).on("progress", function(pe){
        //update progress bar
        if (pe.lengthComputable) {
            var maxsize = pe.total;
            var prog=pe.loaded;
            var perc=prog/maxsize*100;
            perc=perc.toString()+"%"
            progressBar.style.width = perc;
        }
        else {
            fakeProgress += 1;
            var perc=Math.min(100, 100*(1-Math.exp(-fakeProgress/50.)));
            perc=perc.toString()+"%"
            progressBar.style.width = perc;
        }
    }).on("load",function(xhr){
        progressBar.style.width="100%"
        progressParent.classList.add("d-none");
        dataUtils.processRawData(data_id,xhr)
    });
}

/** 
* @param {Object} thecsv csv file path
* This is a function to deal with the request of a csv from a server as opposed to local.
* It creates an XMLHttpRequest. In the future someone should implement it with Fetch to comply with the W3 API
* Its not yet compatible with the new dataUtils.....
*/
dataUtils.XHRCSV = function(data_id, options) {
    console.log(data_id, options, options.path, options["path"]);
    dataUtils.createDataset(data_id,{"name":data_id});
    
    let data_obj = dataUtils.data[data_id];
    data_obj["expectedHeader"] = options.expectedHeader
    data_obj["expectedRadios"] = options.expectedRadios
    data_obj["_csv_path"] = options["path"];
    var op = tmapp["object_prefix"];

    var panel = interfaceUtils.getElementById(data_id+"_input_csv_col");
    panel.classList.add("d-none");

    var xhr = new XMLHttpRequest();

    var progressParent=interfaceUtils.getElementById(data_id+"_csv_progress_parent");
    progressParent.classList.remove("d-none");
    var progressBar=interfaceUtils.getElementById(data_id+"_csv_progress");
    
    var fakeProgress = 0;
    
    // Setup our listener to process compeleted requests
    xhr.onreadystatechange = function () {        
        // Only run if the request is complete
        if (xhr.readyState !== 4) return;        
        // Process our return data
        if (xhr.status >= 200 && xhr.status < 300) {
            // What do when the request is successful
            progressBar.style.width = "100%";
            progressParent.classList.add("d-none");
            console.log(xhr);
            dataUtils.processRawData(data_id,d3.csvParse(xhr.responseText));
        }else{
            console.log("dataUtils.XHRCSV responded with "+xhr.status);
            progressParent.classList.add("d-none");
            interfaceUtils.alert ("Impossible to load data")
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

    xhr.open('GET', options["path"]);
    xhr.send();
    
}

/**
 * Creeate the data_obj[op + "_barcodeGarden"] ("Garden" as opposed to "forest")
 * To save all the trees per barcode or per key. It is an object so that it is easy to just call
 * the right tree given the key. It will be created every time the user wants to group by something different, 
 * replacing the previous garden. It might date some time with big datasets, use at your own discretion.
 */
dataUtils.makeQuadTrees = function(data_id) {

    var data_obj = dataUtils.data[data_id];

    //get x and Y from inputs
    var inputs=interfaceUtils._mGenUIFuncs.getTabDropDowns(data_id);
    var xselector=data_obj["_X"]
    var yselector=data_obj["_Y"]
    var groupByCol=data_obj["_gb_col"]
    var groupByColsName=data_obj["_gb_name"]

    //console.log(xselector,yselector,groupByCol,groupByColsName)

    //little optimization to not redoo the tree if we have it
    /*if(inputs["gb_col"].value==data_obj["_gb_col"]){
        if(Object.keys(data_obj["_groupgarden"]).length > 0){
            message="Group garden exists, dont waste time recreating it";
            //interfaceUtils.alert(message); 
            console.log(message);
            
            return; //because graden exists
        }
    }*/

    var x = function (d) {
        return d[xselector];
    };
    var y = function (d) {
        return d[yselector];
    };
    console.log("groupByCol", groupByCol);
    if (groupByCol) {
        var allgroups = d3.nest().key(function (d) { return d[groupByCol]; }).entries(data_obj["_processeddata"]);

        data_obj["_groupgarden"] = {};
        for (var i = 0; i < allgroups.length; i++) {
            var treeKey = allgroups[i].values[0][groupByCol];
            data_obj["_groupgarden"][treeKey] = d3.quadtree().x(x).y(y).addAll(allgroups[i].values);
            data_obj["_groupgarden"][treeKey]["treeID"] = treeKey; // this is also the key in the groupgarden but just in case
            
            if(groupByColsName){
                var treeName = allgroups[i].values[0][groupByColsName] || "";
                data_obj["_groupgarden"][treeKey]["treeName"] = treeName;
            }
            //create the subsampled for all those that need it
        }
    }
    else {
        console.log("No group, we take everything!");
        treeKey = "All";
        data_obj["_groupgarden"] = {};
        data_obj["_groupgarden"][treeKey] = d3.quadtree().x(x).y(y).addAll(data_obj["_processeddata"]);
        data_obj["_groupgarden"][treeKey]["treeID"] = treeKey; // this is also the key in the groupgarden but just in case
        
        if(groupByColsName){
            var treeName = data_obj["_processeddata"][0][groupByColsName] || "";
            data_obj["_groupgarden"][treeKey]["treeName"] = treeName;
        }
    }
    //print UIs that are only for groups
    //markerUtils.printBarcodeUIs(data_obj._drawOptions);
    
}

/** 
 * @deprecated
 * Take the HTML input and read the file when it is changed, take the data 
 * type and the html dom id
*/
dataUtils.processEventForCSV = function(data_id, dom_id) {
    //the dom id has to be of an input type 
    if(!dom_id.includes("#"))
        dom_id="#"+dom_id
        d3.select(dom_id)
        .on("change", function () {
            var file = d3.event.target.files[0];
            if (file) {
                var reader = new FileReader();
                reader.onloadend = function (evt) {
                    var dataUrl = evt.target.result;
                    dataUtils.readCSV(data_id,dataUrl);
                };
                reader.readAsDataURL(file);
            }
        });
}
