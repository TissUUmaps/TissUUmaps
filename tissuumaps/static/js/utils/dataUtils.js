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

    _quadtreesEnabled: true,    // If false, only generate fake empty trees
    _quadtreesMethod: 2,        // 0: D3 quadtrees; 1: depth-limited; 2: depth-limited (array version)
    _quadtreesMaxDepth: 8,      // Only used for depth-limited trees
    _quadtreesLastInputs: {},  // Store some info to avoid recomputing a quadtree if not necessary
}

/** 
* Creates an object inside dataUtils.data so that all options can be grouped by csv
* @param {String} uid The id of the data group
* @param {Object} options options that could be used inside, only holds name currently
 */
dataUtils.createDataset = function(uid,options){
    if(!options) options={};
    dataUtils.data[uid]={
        _type: "GENERIC_DATA",
        _filetype: options.filetype || "csv",
        _name:options.name || "",
        _processeddata:undefined,
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
* Selects a data object and reads a csv and starts the whole process to add it in datautils and the interface. 
* It can only be associated to a filepicker. It is atomatically listened when created in the interfaceUtils in the change event
* @param {HTMLEvent} event the event arries the file picker. which MUST have the data id in the begining separated by an "_"
*/
dataUtils.startCSVcascade= function(event){
    var data_id=event.target.id.split("_")[0];
    var file = event.target.files[0];
    if (["h5","h5ad"].includes(file.name.split('.').pop() )) {
        dataUtils.readH5(data_id, file);
    }
    else {
        dataUtils.readCSV(data_id, file);
    }
    /*if (file) {
        var reader = new FileReader();
        reader.onloadend = function (evt) {
            var dataUrl = evt.target.result;
            dataUtils.readCSV(data_id,dataUrl);
        };
        reader.readAsDataURL(file);
    }*/
}

/**
 * @deprecated Not required anymore, but kept for backwards-compatibility
 */
CPDataUtils={};

/** 
* created the _processeddata list to be used in rendering
* @param {String} data_id The id of the data group like "U234345"
* @param {Array} data data coming from d3 after parsing the csv
*/
dataUtils.processRawData = function(data_id, rawdata) {
    let data_obj = dataUtils.data[data_id];

    data_obj["_processeddata"].columns = rawdata.columns;
    for (let i = 0; i < rawdata.columns.length; ++i) {
        // Convert chunks of column into a single large array
        if (rawdata.isnan[i]) {
            data_obj["_processeddata"][rawdata.columns[i]] = rawdata.data[i].flat();
        } else {
            const numRows = rawdata.data[i].reduce((x, y) => x + y.length, 0);
            data_obj["_processeddata"][rawdata.columns[i]] = new Float64Array(numRows);

            let offset = 0;
            for (chunk of rawdata.data[i]) {
                data_obj["_processeddata"][rawdata.columns[i]].set(chunk, offset);
                offset += chunk.length;
            }
        }
        delete rawdata.data[i];  // Clean up memory
    }

    //this function is in case we need to standardize the data column names somehow,
    //so that the processseddata has some desired structure, but for now maybe no

    dataUtils.createMenuFromCSV(data_id, rawdata.columns);

}

dataUtils.getAllH5Data = function(data_id, alldrops){
    var data_obj = dataUtils.data[data_id];

    let getH5Data = function(drop){
        return new Promise((resolve, reject) => {
            if (!alldrops[drop]) {reject(null);return}
            if (alldrops[drop].value != "") {
                if (data_obj["_processeddata"].columns.includes(alldrops[drop].value)) {
                    resolve(drop);return
                }
                let h5paths = alldrops[drop].value.split(";");
                let h5range = 0;
                let h5path = h5paths[0];
                if (h5paths.length > 1) {
                    h5range = h5paths[1];
                }
                data_obj["_processeddata"].columns.push(alldrops[drop].value);
                data_obj["_csv_header"].push(alldrops[drop].value);
                let url = data_obj["_csv_path"];
                dataUtils._hdf5Api.getXRow(url, h5range, h5path).then((data) => {
                    if (Object.prototype.toString.call(data).includes("BigInt") || Object.prototype.toString.call(data).includes("BigUint")) {
                        data = [...data].map((x)=>Number(x));
                    }
                    data_obj["_processeddata"][alldrops[drop].value] = data;
                    resolve(drop);return
                },
                (error) => {
                    reject(error);return 
                })
            }
            else {
                reject(null);
            }
        })
    }
    if (alldrops === undefined) {
        var alldrops=interfaceUtils._mGenUIFuncs.getTabDropDowns(data_id, true);
    }
    var namesymbols=Object.getOwnPropertyNames(alldrops);
    namesymbols = namesymbols.filter((val)=>{return alldrops[val].value != ""})
    // TODO: keep columns if needed!
    console.log(data_obj["_processeddata"])
    if (data_obj["_processeddata"] === undefined) {
        data_obj["_processeddata"] = {
            columns : []
        };
        data_obj["_csv_header"] = [];
    }
    let progressBar=interfaceUtils.getElementById(data_id+"_csv_progress");
    progressBar.style.width = "10%";
    // We get H5 data for each field, sequentially:
    return namesymbols.reduce(function(p, drop, drop_index) {
        return p.then(function(results) {
            return getH5Data(drop).then(function(data) {
                let perc=100 * (drop_index+1) / namesymbols.length;
                perc=perc.toString()+"%";
                console.log(drop_index, namesymbols.length,perc);
                progressBar.style.width = perc;
                results.push(data);
                return results;
            },
            function(error) {return results});
        });
    }, Promise.resolve([]));
}

/** 
* Make sure that the options selected are correct an call the necessary functions to process the data so
* its ready to be displayed.
* @param {String} data_id The id of the data group like "U234345"
*/
dataUtils.updateViewOptions = async function(data_id, force_reload_all, reloadH5){
    if (reloadH5 === undefined) reloadH5 = true;
    let progressParent=interfaceUtils.getElementById(data_id+"_csv_progress_parent");
    progressParent.classList.remove("d-none");
    let progressBar=interfaceUtils.getElementById(data_id+"_csv_progress");
    progressBar.style.width = "10%";
    
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

    var updateButton = document.getElementById(data_id + "_update-view-button")
    updateButton.innerHTML = "Loading..."
    
    var p = Promise.resolve();
    if (data_obj._filetype == "h5" && reloadH5) {
        p = await dataUtils.getAllH5Data(data_id);
    }

    if(inputs["X"].value == 'null' || inputs["Y"].value == 'null'){
        message="Select X and Y first";
        interfaceUtils.alert(message); console.log(message);
        return;
    }else{
        data_obj["_X"]=inputs["X"].value;
        data_obj["_Y"]=inputs["Y"].value;
    }
    // Check if image is already fake:
    var recompute_background_img = false;
    if (tmapp["ISS_viewer"].world.getItemCount() == 0) {
        recompute_background_img = true;
    }
    else {
        if (tmapp["ISS_viewer"].world.getItemAt(0).source.getTileUrl(0,0,0) == null) {
            var recompute_background_img = true;
        }
    }
    if (recompute_background_img) {
        function getMax(arr) {
            let len = arr.length; let max = -Infinity;
            while (len--) { max = +arr[len] > max ? +arr[len] : max; }
            return max;
        }
        function getMin(arr) {
            let len = arr.length; let min = Infinity; 
            while (len--) { min = +arr[len] < min ? +arr[len] : min; }
            return min;
        }
        var minX = getMin(data_obj["_processeddata"][data_obj["_X"]]);
        var maxX = getMax(data_obj["_processeddata"][data_obj["_X"]]);
        var minY = getMin(data_obj["_processeddata"][data_obj["_Y"]]);
        var maxY = getMax(data_obj["_processeddata"][data_obj["_Y"]]);
        if (minX <0 || maxX < 500 || minY <0 || maxY < 500) {
            var markerTransform;
            if (maxX - minX > maxY - minY) {
                markerTransform = 5000 / (maxX - minX);
            }
            else {
                markerTransform = 5000 / (maxY - minY);
            }
            let arrX = data_obj["_processeddata"][data_obj["_X"]];
            for (let i = 0; i < arrX.length; ++i) {
                arrX[i] = markerTransform * (arrX[i] - minX);
            }
            maxX = getMax(arrX);
            let arrY = data_obj["_processeddata"][data_obj["_Y"]];
            for (let i = 0; i < arrY.length; ++i) {
                arrY[i] = markerTransform * (arrY[i] - minY);
            }
            maxY = getMax(arrY);
        }
        // We load an empty image at the size of the data.
        if (tmapp["ISS_viewer"].world.getItemCount() > 0) {
            if (tmapp["ISS_viewer"].world.getItemAt(0).source.height < parseInt(maxY*1.06) || tmapp["ISS_viewer"].world.getItemAt(0).source.width < parseInt(maxX*1.06)){
                tmapp["ISS_viewer"].close();
                setTimeout (function() {dataUtils.updateViewOptions(data_id, false, false)},50);
                return;
            }
        }
        else {
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
            setTimeout (function() {dataUtils.updateViewOptions(data_id, true, false)},50);
            return;
        }
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
        data_obj["_cb_gr_dict"]=inputs["cb_gr_dict"].value;
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
    data_obj["_pie_dict"]=inputs["pie_dict"].value;
    if (data_obj["_pie_col"]=="null") {
        interfaceUtils.alert("No piechart column selected. Impossible to update view.");return;
    }
    data_obj["_edges_col"]=(radios["edges_check"].checked ? inputs["edges_col"].value : null);
    if (data_obj["_edges_col"]=="null") {
        interfaceUtils.alert("No edges column selected. Impossible to update view.");return;
    }
    data_obj["_sortby_col"]=(radios["sortby_check"].checked ? inputs["sortby_col"].value : null);
    data_obj["_sortby_desc"]=(radios["sortby_check"].checked ? radios["sortby_desc_check"].checked : null);
    data_obj["_z_order"]=parseFloat(inputs["z_order"].value);

    if (data_obj["_sortby_col"]=="null") {
        interfaceUtils.alert("No sort by column selected. Impossible to update view.");return;
    }
    // Use scale colummn
    data_obj["_scale_col"]=(radios["scale_check"].checked ? inputs["scale_col"].value : null);
    if (data_obj["_scale_col"]=="null") {
        interfaceUtils.alert("No size column selected. Impossible to update view.");return;
    }
    data_obj["_scale_factor"]=parseFloat(inputs["scale_factor"].value);
    data_obj["_coord_factor"]=parseFloat(inputs["coord_factor"].value);
    // Use shape column
    data_obj["_shape_col"]=(radios["shape_col"].checked ? inputs["shape_col"].value : null);
    if (data_obj["_shape_col"]=="null") {
        interfaceUtils.alert("No shape column selected. Impossible to update view.");return;
    }
    // Use opacity column
    data_obj["_opacity_col"]=(radios["opacity_check"].checked ? inputs["opacity_col"].value : null);
    if (data_obj["_opacity_col"]=="null") {
        interfaceUtils.alert("No opacity column selected. Impossible to update view.");return;
    }// Use collection column
    data_obj["_collectionItem_col"]=(radios["collectionItem_col"].checked ? inputs["collectionItem_col"].value : null);
    data_obj["_collectionItem_fixed"]=(radios["collectionItem_col"].checked ? null : parseInt(inputs["collectionItem_fixed"].value));
    if (data_obj["_collectionItem_col"]=="null") {
        interfaceUtils.alert("No collection item column selected. Impossible to update view.");return;
    }
    if (
        (data_obj["_collectionItem_col"] || data_obj["_collectionItem_fixed"] > 0)
        && filterUtils._compositeMode != "collection") {
        //interfaceUtils.alert("Warning, images are not in Collection Mode. Go to \"Image layers > Filter Settings > Merging mode\" to activate Collection Mode.");
    }
    data_obj["_opacity"]=parseFloat(inputs["opacity"].value);
    // Tooltip
    data_obj["_tooltip_fmt"]=inputs["tooltip_fmt"].value;
    
    data_obj["_no_outline"]=(radios["_no_outline"].checked ? true : false);

    //this function veryfies if a tree with these features exist and doesnt recreate it
    dataUtils.makeQuadTrees(data_id);
    //print a menu in the interface for the groups
    let table=await interfaceUtils._mGenUIFuncs.groupUI(data_id);
    if (table == null) return;
    menuui=interfaceUtils.getElementById(data_id+"_menu-UI");
    menuui.classList.remove("d-none")
    menuui.innerText="";

    menuui.appendChild(table);
    //shape UXXXX_grname_shape, color UXXXX_grname_color

    // Make sure that slider for global marker size is shown
    if (interfaceUtils.getElementById("ISS_globalmarkersize"))
        interfaceUtils.getElementById("ISS_globalmarkersize").classList.remove("d-none");

    if (data_obj["fromButton"] !== undefined) {
        projectUtils.updateMarkerButton(data_id);
    }
    // If we need to reload all markers from all datasets after new image size:
    if(force_reload_all !== undefined) {
        for (var uid in dataUtils.data) {
            glUtils.loadMarkers(uid);
        }
    }
    else {
        glUtils.loadMarkers(data_id);
    }
    glUtils.draw();
    
    setTimeout(()=>{
        // We apply constraints after 300 ms in case the viewport size changed.
        tmapp.ISS_viewer.viewport.applyConstraints(false);
    },300);
    // Create the event.
    const event = document.createEvent('Event');
    // Define that the event name is 'glUtilsDraw'.
    event.initEvent('glUtilsDraw', true, true);
    // target can be any Element or other EventTarget.
    document.dispatchEvent(event);
    updateButton.innerHTML = "Update view"
    progressBar.style.width = "100%";
    progressParent.classList.add("d-none");
}

/** 
* Fills the necessary input dropdowns with the csv headers so that user can choose them
* @param {String} data_id The id of the data group like "U234345"
* @param {Object} datumExample example datum that contains the headers of the csv
*/
dataUtils.createMenuFromCSV = function(data_id,datumExample) {
    var data_obj = dataUtils.data[data_id];

    //var csvheaders = Object.keys(datumExample);
    var csvheaders = datumExample;
    data_obj["_csv_header"] = csvheaders;

    //fill dropdowns
    var alldrops=interfaceUtils._mGenUIFuncs.getTabDropDowns(data_id, true);
    var namesymbols=Object.getOwnPropertyNames(alldrops);
    namesymbols.forEach((drop)=>{
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
* Calls dataUtils.createDataset and loads and parses the csv using D3. 
* then calls dataUtils.createMenuFromCSV to modify the interface in its own tab
* @param {String} data_id The id of the data group like "U234345"
* @param {Object} thecsv csv file path
*/
dataUtils.readH5 = function(data_id, thecsv, options) { 
    interfaceUtils._mGenUIFuncs.dataTabUIToH5(data_id);
    dataUtils.createDataset(data_id,{"name":data_id, "filetype":"h5"});

    let data_obj = dataUtils.data[data_id];
    data_obj["modified"] = true;
    data_obj["_processeddata"] = undefined;
    data_obj["_isnan"] = {};
    data_obj["_csv_header"] = null;
    data_obj["_csv_path"] = thecsv;
    if (options != undefined) {
        //data_obj["_csv_path"] = options.path;
        data_obj["expectedHeader"] = options.expectedHeader;
        data_obj["expectedRadios"] = options.expectedRadios;
        data_obj["fromButton"] = options.fromButton;
        // Hide download button?
        let panel = interfaceUtils.getElementById(data_id+"_input_csv_col");
        panel.classList.add("d-none");
    }
    
    let progressParent=interfaceUtils.getElementById(data_id+"_csv_progress_parent");
    progressParent.classList.remove("d-none");
    let progressBar=interfaceUtils.getElementById(data_id+"_csv_progress");
    progressBar.style.width = "0%";
    
    let url = thecsv;
    dataUtils._hdf5Api.get(url,{path:"/"}).then((data) => {
        progressBar.style.width = "100%";
        progressParent.classList.add("d-none");
        dataUtils._quadtreesLastInputs = {};  // Clear to make sure quadtrees are generated
        if (data_obj["expectedHeader"]) {
            interfaceUtils._mGenUIFuncs.fillRadiosAndChecksIfExpectedCSV(data_id,data_obj["expectedRadios"]);
            interfaceUtils._mGenUIFuncs.fillDropDownsIfExpectedCSV(data_id,data_obj["expectedHeader"]);
            dataUtils.updateViewOptions(data_id);
        }
    })
}


dataUtils.getPath = function () {
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const path = urlParams.get('path')
    return path;
}

/** 
* Calls dataUtils.createDataset and loads and parses the csv using D3. 
* then calls dataUtils.createMenuFromCSV to modify the interface in its own tab
* @param {String} data_id The id of the data group like "U234345"
* @param {Object} thecsv csv file path
*/
dataUtils.readCSV = function(data_id, thecsv, options) { 
    if (dataUtils.data[data_id] === undefined){
        dataUtils.createDataset(data_id,{"name":data_id, "filetype":"csv"});
    }
    let data_obj = dataUtils.data[data_id];
    
    let skip_download = (data_obj["_csv_path"] == options.path);

    data_obj["modified"] = true;
    if (!skip_download) {
        data_obj["_processeddata"] = {};
        data_obj["_isnan"] = {};
        data_obj["_csv_header"] = null;
    }
    data_obj["_csv_path"] = thecsv;
    if (options != undefined) {
        data_obj["_csv_path"] = options.path;
        data_obj["expectedHeader"] = options.expectedHeader;
        data_obj["expectedRadios"] = options.expectedRadios;
        data_obj["fromButton"] = options.fromButton;
        // Hide download button?
        let panel = interfaceUtils.getElementById(data_id+"_input_csv_col");
        panel.classList.add("d-none");
    }
    if (skip_download) {
        let columns = data_obj["_processeddata"].columns;
        dataUtils.createMenuFromCSV(data_id, columns);
        return;
    }
    let progressParent=interfaceUtils.getElementById(data_id+"_csv_progress_parent");
    progressParent.classList.remove("d-none");
    let progressBar=interfaceUtils.getElementById(data_id+"_csv_progress");
    let fakeProgress = 0;
    function getFileSize(url)
    {
      var fileSize = '';
      var http = new XMLHttpRequest();
      http.open('HEAD', url, false); // false = Synchronous
  
      http.send(null); // it will stop here until this http request is complete
  
      // when we are here, we already have a response, b/c we used Synchronous XHR
  
      if (http.status === 200) {
          fileSize = http.getResponseHeader('content-length');
      }
  
      return fileSize;
    }
    var totalSize = undefined;
    if (options != undefined) {
        totalSize = getFileSize(thecsv);
    }
    let updateProgressBar = function(op, progress) {
        if (op == "progress") {
            if (totalSize == undefined) {
                fakeProgress += 1;
                let perc=Math.min(100, 100*(1-Math.exp(-fakeProgress/100.)));
                perc=perc.toString()+"%";
                progressBar.style.width = perc;
            }
            else {
                var perc= Math.round(progress / totalSize * 100);
                perc=perc.toString()+"%";
                progressBar.style.width = perc;
            }
        }
        if (op == "load") {
            // Hide progress bar
            progressBar.style.width="100%";
            progressParent.classList.add("d-none");
        }
    };
    
    let rawdata = { columns: [], isnan: [], data: [], tmp: [] };
    console.time("Load CSV");
    Papa.parse(thecsv, {
        download: (options != undefined),
        delimiter: ",",
        header: false,
   	    worker: false,
        step: function(row) {
            if (rawdata.columns.length == 0) {
                const header = row.data;
                for (let i = 0; i < header.length; ++i) {
                    rawdata.columns[i] = header[i];
                    rawdata.isnan[i] = false;
                    rawdata.data[i] = [];
                }
                rawdata.tmp = rawdata.columns.map(x => []);
            } else {
                // Check so that we are not processing an incomplete row
                if (row.data.length != rawdata.columns.length) return;

                for (let i = 0; i < row.data.length; ++i) {
                    const value = row.data[i];
                    // Update type flag of column and push value to temporary buffer
                    rawdata.isnan[i] = rawdata.isnan[i] || isNaN(value) || (value == "");
                    rawdata.tmp[i].push(rawdata.isnan[i] ? value : +value);
                }
                if (rawdata.tmp[0].length >= 10000) {
                    // Push content of temporary buffers to output arrays
                    for (let i = 0; i < rawdata.columns.length; ++i) {
                        rawdata.data[i].push(rawdata.isnan[i] ? rawdata.tmp[i]
                                                              : new Float64Array(rawdata.tmp[i]));
                    }
                    rawdata.tmp = rawdata.columns.map(x => []);  // Clear buffers
                    updateProgressBar("progress", row.meta.cursor);
                }
            }
        },
        complete: function(result) {
            if (rawdata.tmp.length > 0 && rawdata.tmp[0].length > 0) {
                // Push content of temporary buffers to output arrays
                for (let i = 0; i < rawdata.columns.length; ++i) {
                    rawdata.data[i].push(rawdata.isnan[i] ? rawdata.tmp[i]
                                                          : new Float64Array(rawdata.tmp[i]));
                }
                rawdata.tmp = rawdata.columns.map(x => []);  // Clear buffers
            }
            updateProgressBar("load");
            console.timeEnd("Load CSV");
            dataUtils._quadtreesLastInputs = {};  // Clear to make sure quadtrees are generated
            dataUtils.processRawData(data_id, rawdata);
        },
        error: function() {
            interfaceUtils.alert("Impossible to load csv file, please check relative path in the tmap file:<br/><code>" + thecsv + "</code>","CSV loading error...");
            interfaceUtils._mGenUIFuncs.deleteTab(data_id);
        }
    });
}

/** 
* This is a function to deal with the request of a csv from a server as opposed to local.
* @param {Object} thecsv csv file path
*/
dataUtils.XHRCSV = function(data_id, options) {
    var csvFile = options["path"];
    const path = dataUtils.getPath();
    if (path != null) {
        csvFile = path + "/" + csvFile;
    }
    if (["h5","h5ad"].includes(csvFile.split('.').pop() )) {
        dataUtils.readH5(data_id, csvFile, options);
    }
    else {
        dataUtils.readCSV(data_id, csvFile, options);
    }
}

/**
 * Create the data_obj[op + "_barcodeGarden"] ("Garden" as opposed to "forest")
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
    var markerData=data_obj["_processeddata"];

    // Check if we can skip recomputing the last generated quadtree
    const lastInputs = dataUtils._quadtreesLastInputs;
    const newInputs = {
        "uid": data_id, "_X": xselector, "_Y": yselector,
        "_gb_col": groupByCol, "_gb_name": groupByColsName,
        "_quadtreesEnabled": dataUtils._quadtreesEnabled,
        "_quadtreesMaxDepth": dataUtils._quadtreesMaxDepth,
        "_quadtreesMethod": dataUtils._quadtreesMethod,
    };
    if (JSON.stringify(lastInputs) == JSON.stringify(newInputs)) return;  // Nothing more to do!
    dataUtils._quadtreesLastInputs = newInputs;

    const numMarkers = markerData[xselector].length + 0;
    let indexData = new Uint32Array(numMarkers);
    for (let i = 0; i < numMarkers; ++i) indexData[i] = i;

    var x = function (d) {
        return markerData[xselector][d];
    };
    var y = function (d) {
        return markerData[yselector][d];
    };
    if (dataUtils._quadtreesEnabled) console.time("Generate quadtrees");
    if (groupByCol) {
        var allgroups = d3.nest().key(function (d) { return markerData[groupByCol][d]; }).entries(indexData);

        data_obj["_groupgarden"] = {};
        for (var i = 0; i < allgroups.length; i++) {
            const treeKey = allgroups[i].key;
            if (dataUtils._quadtreesEnabled) {
                allgroups[i].values = new Uint32Array(allgroups[i].values);
                if (dataUtils._quadtreesMethod == 0) {
                    data_obj["_groupgarden"][treeKey] = d3.quadtree().x(x).y(y).addAll(allgroups[i].values);
                } else {
                    const maxDepth = dataUtils._quadtreesMaxDepth;
                    const useArrayLeaves = dataUtils._quadtreesMethod == 2;
                    data_obj["_groupgarden"][treeKey] = d3.quadtree().x(x).y(y);
                    dataUtils._quadtreeAddAll(data_obj["_groupgarden"][treeKey], allgroups[i].values, maxDepth, useArrayLeaves);
                }
            } else {
                const groupSize = allgroups[i].values.length + 0;
                data_obj["_groupgarden"][treeKey] = {"size" : function() { return groupSize; }};
            }
            data_obj["_groupgarden"][treeKey]["treeID"] = treeKey; // this is also the key in the groupgarden but just in case
            
            if (groupByColsName) {
                const treeName = data_obj["_processeddata"][groupByColsName][allgroups[i].values[0]] || "";
                data_obj["_groupgarden"][treeKey]["treeName"] = treeName;
            }
        }
    }
    else {
        console.log("No group, we take everything!");
        treeKey = "All";
        data_obj["_groupgarden"] = {};
        if (dataUtils._quadtreesEnabled) {
            if (dataUtils._quadtreesMethod == 0) {
                data_obj["_groupgarden"][treeKey] = d3.quadtree().x(x).y(y).addAll(indexData);
            } else {
                const maxDepth = dataUtils._quadtreesMaxDepth;
                const useArrayLeaves = dataUtils._quadtreesMethod == 2;
                data_obj["_groupgarden"][treeKey] = d3.quadtree().x(x).y(y);
                dataUtils._quadtreeAddAll(data_obj["_groupgarden"][treeKey], indexData, maxDepth, useArrayLeaves);
            }
        } else {
            data_obj["_groupgarden"][treeKey] = {"size" : function() { return numMarkers; }};
        }
        data_obj["_groupgarden"][treeKey]["treeID"] = treeKey; // this is also the key in the groupgarden but just in case
        
        if (groupByColsName) {
            const treeName = data_obj["_processeddata"][groupByColsName][0] || "";
            data_obj["_groupgarden"][treeKey]["treeName"] = treeName;
        }
    }
    if (dataUtils._quadtreesEnabled) console.timeEnd("Generate quadtrees");
}

/**
 * Helper function for dataUtils._quadTreeAddAll(), and should therefore not be
 * called directly outside of that function.
 */
dataUtils._quadtreeAdd = function(tree, x, y, d, maxDepth, useArrayLeaves) {
    if (isNaN(x) || isNaN(y)) return;  // Ignore invalid points

    let parent,
        node = tree._root,
        leaf = {data: d},
        x0 = tree._x0, y0 = tree._y0,
        x1 = tree._x1, y1 = tree._y1,
        xm, ym, xp, yp,
        right, bottom, i;

    // If the tree is empty, initialize the root
    if (!node) {
        node = tree._root = new Array(4);
    }

    // Find leaf node location at maxDepth level and allocate new nodes
    // for the path in the tree
    for (let depth = 0; depth < maxDepth; ++depth) {
        if (right = x >= (xm = (x0 + x1) / 2)) x0 = xm; else x1 = xm;
        if (bottom = y >= (ym = (y0 + y1) / 2)) y0 = ym; else y1 = ym;
        i = bottom << 1 | right;

        if (depth < (maxDepth - 1) && !node[i]) {
            // Allocate new node
            node[i] = new Array(4);
        }
        parent = node, node = node[i];
    }

    if (useArrayLeaves) {
        // Insert point into leaf node's data array
        parent[i] = !parent[i] ? {data: []} : parent[i];
        parent[i].data.push(leaf.data);
    } else {
        // Insert point into linked list of leaf nodes
        leaf.next = node;
        parent[i] = leaf;
    }
}

/**
 * Generate a tree of a fixed depth for better memory efficiency when used with
 * large point datasets. Use instead of d3.quadtree.addAll().
 */
dataUtils._quadtreeAddAll = function(tree, indices, maxDepth, useArrayLeaves) {
    const n = indices.length;
    let x0 = Infinity, y0 = x0, x1 = -x0, y1 = x1;

    // Compute the points and their extent
    for (let i = 0, d, x, y; i < n; ++i) {
        if (isNaN(x = +tree._x.call(null, d = indices[i])) || isNaN(y = +tree._y.call(null, d))) continue;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
    }

    // If there were no (valid) points, abort
    if (x0 > x1 || y0 > y1) return tree;

    // Expand the tree to cover the new points
    tree.cover(x0, y0).cover(x1, y1);

    // Allocate nodes for depth limited tree and insert points at leaf level
    for (let i = 0; i < n; ++i) {
        const d = indices[i];
        const x = +tree._x.call(null, d);
        const y = +tree._y.call(null, d);
        dataUtils._quadtreeAdd(tree, x, y, d, maxDepth, useArrayLeaves);
    }
    return tree;
}

/**
 * Get the number of points in the tree. Use instead of d3.quadtree.size().
 */
dataUtils._quadtreeSize = function(tree) {
    //console.time("Get quadtree size");
    let size = 0;
    if (dataUtils._quadtreesEnabled && dataUtils._quadtreesMethod == 2) {
        tree.visit(function(node) {
            if (!node.length) do size += node.data.length; while (node = node.next)
        });
    } else {
        size = tree.size();
    }
    //console.timeEnd("Get quadtree size");
    return size;
}
