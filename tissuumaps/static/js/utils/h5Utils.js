/**
* @file h5Utils.js Utilities for h5-based marker loading
* @author Christophe Avenel
* @see {@link h5Utils}
*/

/**
 * @namespace h5Utils
 * @property {Boolean} _initialized True when h5Utils has been initialized
 */
 h5Utils = {
    relative_root: './'
 }

class H5_API {
    constructor() {
        this.chunkSize = 5 * 1024 * 1024;
        this.resolvers = {};
        this.count = 0; // used later to generate unique ids
        this.status = {}

        this.worker = new Worker(URL.createObjectURL(new Blob(["("+worker_function.toString()+")()"], {type: 'text/javascript'})));
        this.worker.addEventListener('message', (e) => {
            console.log("Received:", e);
            let data = e.data;
            let id   = e.data["id"];
            this.resolvers[ id ](data);
            delete this.resolvers[id]; // Prevent memory leak
        });
    }
  
    loadPromise (url) {
        let requestChunkSize = this.chunkSize;
        const id = this.count++;
        let _url = url;
        if (typeof url === 'string' || url instanceof String)
            if (!_url.startsWith("https")) 
                _url = window.location.origin + "/" + _url;
        this.worker.postMessage({id: id, action: "load", payload: {requestChunkSize, url:_url}});
        return new Promise(resolve => this.resolvers[id] = resolve);
    }

    load (url) {
        if (typeof url === 'string' || url instanceof String) {
            var urlName = url;
        }
        else {
            var urlName = "file:" + url.name;
        }
        this.status[urlName] = "loading";
        this.loadPromise(url).then((data)=>{
            if (data.type == "success") {
                setTimeout(()=>{this.status[urlName] = "loaded";},50);
            }
            else {
                this.status[urlName] = "failed";
            }
        })
    }

    get (url, payload, action) {
        if (action === undefined) action = "get"; 
        function sleep (time) {
            return new Promise((resolve) => setTimeout(resolve, time));
        }
        if (!(typeof url === 'string' || url instanceof String)) {
            var urlName = "file:" + url.name;
        }
        else {
            var urlName = url;
        }
        console.log("get:",url, urlName, this.status[urlName]);
        if (this.status[urlName] === undefined) {
            this.load(url);
        }
        return new Promise((resolve, reject) => {
            if (this.status[urlName] === "failed") {
                reject("Impossible to load data");
            }
            if (this.status[urlName] === "loading") {
                sleep(50).then (()=>{
                    this.get(url, payload, action).then((data)=>{
                        resolve(data)
                    })
                    .catch((data)=>{
                        reject("Impossible to load data")
                    })
                });
                return;
            }
            const id = this.count++;
            this.resolvers[id] = resolve
            if (typeof url === 'string' || url instanceof String) {
                if (!url.startsWith("https")) {
                    payload.url = window.location.origin + "/" + url;
                }
                else {
                    payload.url = url;
                }
            }
            else {
                payload.url = urlName;
            }
            console.log(payload, action);
            this.worker.postMessage({id: id, action: action, payload: payload});
        });
    }
}

class H5AD_API  extends H5_API {
        
    getX_join (url, rowIndex, path) {
        return new Promise(resolve => {
            this.get(url,{path:path+"/categories"}).then((data_categ) => {
                this.get(url,{path:path+"/codes"}).then((data_codes) => {
                    const row = [...data_codes.value].map((x)=>data_categ.value[x]);
                    resolve(row);
                });
            });
        });
    }
        
    getXRow_categ (url, rowIndex, path) {
        return new Promise(resolve => {
            this.get(url,{path:path+"/categories"}).then((data_categ) => {
                this.get(url,{path:path+"/codes"}).then((data_codes) => {
                    const row = [...data_codes.value].map((x)=>data_categ.value[x]);
                    resolve(row);
                });
            });
        });
    }
        
    getXRow_csc (url, rowIndex, path) {
        return new Promise(resolve => {
            this.get(url,{path:path}, "attr").then((data_X) => {
                var rowLength = Number(data_X.attrs["shape"][0]);
                this.get(url,{path:path+"/indptr"}).then((indptr) => {
                    let x1 = indptr.value[parseInt(rowIndex)];
                    let x2 = indptr.value[parseInt(rowIndex)+1];
                    this.get(url,{path:path+"/indices", slice:[[x1,x2]]}).then((indices) => {
                        this.get(url,{path:path+"/data", slice:[[x1,x2]]}).then((data) => {
                            const row = new Float32Array(rowLength);
                            for (let i=0; i<indices.value.length;i++) {
                                row[indices.value[i]] = data.value[i];
                            }
                            resolve(row);
                        });
                    });
                });
            });
        });
    }

    async getXRow_csr (url, colIndex, path) {
        colIndex = parseInt(colIndex);
        let indptr = await this.get(url,{path:path+"/indptr"});
        indptr = indptr.value;
        // Determine the number of rows in the CSR matrix
        const numRows = indptr.length - 1;
        console.log(numRows, indptr);
        // Create a typed array to store the column values
        const columnValues = new Float32Array(numRows);
        
        // Loop through the row indices
        for (let i = 0; i < numRows; i++) {
            if (i%10 == 0) console.log(i, numRows);
            const rowStart = indptr[i];
            const rowEnd = indptr[i + 1];
        
            // Use binary search to find the desired column in the current row
            let low = rowStart;
            let high = rowEnd - 1;
        
            while (low <= high) {
                if (low > high || low < rowStart || high > rowEnd) {
                    break;
                }
                if (high - low < 5000) {
                    let cols = await this.get(url,{path:path+"/indices", slice:[[low,high+1]]})
                    let col_i = cols.value.indexOf(colIndex);
                    if (col_i == -1) break;
                    let value = await this.get(url,{path:path+"/data", slice:[[low+col_i,low+col_i+1]]})
                    columnValues[i] = value.value[0];
                    break;
                }
                const mid = Math.floor((low + high) / 2);
                let col = await this.get(url,{path:path+"/indices", slice:[[mid,mid+1]]})
                col = col.value[0];
                
                if (col == colIndex) {
                    // If the current value belongs to the desired column, store it
                    let value = await this.get(url,{path:path+"/data", slice:[[mid,mid+1]]})
                    columnValues[i] = value.value[0];
                    break;
                } else if (col > colIndex) {
                    // If the current column is less than the desired column, search in the right half
                    low = mid + 1;
                } else {
                    // If the current column is greater than the desired column, search in the left half
                    high = mid - 1;
                }
            }
        }
        
        // Return the column values as a typed array
        return columnValues;
    }
        
    getXRow_array (url, rowIndex, path) {
        return new Promise(resolve => {
            this.get(url,{path:path, slice:[[],[parseInt(rowIndex), parseInt(rowIndex)+1]]}).then((data_X) => {
                resolve(data_X.value);
            });
        });
    }

    getXRow (url, rowIndex, path) {
        return new Promise(resolve => {
            this.get(url,{path:path}, "attr").then((data_X) => {
                if (rowIndex == "join") {
                    this.get (url, {path:path+"/indptr"}).then((indptr)=>{
                        this.get (url, {path:path+"/indices"}).then((indices)=>{
                            var str_array = [];
                            
                            for (let i=0; i<indptr.value.length;i++) {
                                str_array.push(
                                    indices.value.slice(indptr.value[i], indptr.value[i+1]).join(";")
                                );
                            }
                            resolve(str_array);
                        });
                    });
                    return;
                }
                if (data_X.attrs === undefined) {
                    interfaceUtils.alert("data_X.attrs === undefined")
                    return this.getXRow_array (url, rowIndex, path).then((data)=>{
                        resolve(data);
                    });
                }
                
                if (data_X.attrs["encoding-type"] == "categorical") {
                    this.getXRow_categ (url, rowIndex, path).then((data)=>{
                        resolve(data);
                    });
                }
                else if (data_X.attrs["encoding-type"] == "csc_matrix") {
                    this.getXRow_csc (url, rowIndex, path).then((data)=>{
                        resolve(data);
                    });
                }
                else if (data_X.attrs["encoding-type"] == "csr_matrix") {
                    interfaceUtils.alert("CSR sparse format will be slow to read, please convert your data to CSC by using:<br/><code>st_adata.X = scipy.sparse.csc_matrix(st_adata.X)</code>")
                    this.getXRow_csr (url, rowIndex, path).then((data)=>{
                        resolve(data);
                    });
                    //resolve("csr sparse format not supported!")
                }
                else {
                    return this.getXRow_array (url, rowIndex, path).then((data)=>{
                        resolve(data);
                    });
                }
            });
        });
    }
    
    getKeys (url, path, checkBack) {
        if (checkBack === undefined) checkBack = true;
        if (path === undefined) path = "/";
        if (path[0] != "/") path = "/" + path;
        return new Promise((resolve, reject) => {
            this.get(url,{path:path}, "keys").then((data_keys) => {
                if (data_keys.type == "Dataset") {
                    let children = [];
                    if (data_keys.shape.length > 1) {
                        for (let i=0; i<data_keys.shape[1];i++){
                            children.push(path+";"+i.toString());
                        }
                    }
                    resolve({children:children});
                }
                else if (data_keys.type == "Group") {
                    resolve(data_keys);
                }
                else if (checkBack){
                    path = path.substring(0, Math.max(path.lastIndexOf('/'),path.lastIndexOf(';')));
                    this.getKeys(url, path).then((data_keys_root)=>{
                        resolve(data_keys_root);
                    })
                }
                else {
                    resolve ({children:[]})
                }
            })
            .catch((err) => {
                reject(err);
            });
        });
    }

    async getKeysNames (url, path) {
        try {
            let keys = await this.getKeys (url, path, false);
            console.log(keys);
            return keys.children.map((x) => {
                return /[^/]*$/.exec(x)[0];
            });
        }
        catch (err) {
            throw (err);
        }
    }
}
/*
var hdf5Api = new H5AD_API()
let url = "/scANVI_kidney_object.h5ad";//"/adata_msbrain_3rep_withclusters_csc.h5ad";
hdf5Api.getKeys(url).then((data) => {
    console.log(data);
})*/
/*
hdf5Api.getXRow(url, 6, "X").then((data) => {
    console.log(data);
})
hdf5Api.loadPromise(url).then((data) => {
    console.log(data);
});*/

// Genes:   var/_index
// globalX: obsm/spatial;0
// globalY: obsm/spatial;1
// Num Obs: obs/*
// Cat Obs: obs/*/codes + obs/*/categories


var tmap_template = {
    "layers": [{"name": "tissue.tif", "tileSource": "./img/tissue.tif.dzi"}],
    "markerFiles": [],
    "plugins": [],
    "collectionMode": true,
}


async function getPalette(h5url, obs) {
    let palette = {};
    try {
        let obsCategories = await dataUtils._hdf5Api.get(h5url, {path:"/obs/" + obs + "/categories"});
        if (obsCategories.type == "error") return {};
        let unsColors = await dataUtils._hdf5Api.get(h5url, {path:"/uns/" + obs + "_colors"});
        if (unsColors.type == "error") return {};
        let newPalette = Object.fromEntries(obsCategories.value.map((x, i) => [x, unsColors.value[i].slice(0, 7)]));
        palette = {...palette, ...newPalette};
    } catch (error) {
        // pass
        alert(error);
    }
    return palette;
}

async function getVarList(h5url) {
    let varList;
    try {
        varList = await dataUtils._hdf5Api.get(h5url, {path:"/var/_index"});
        if (varList.type == "error") {
            let _var = await dataUtils._hdf5Api.get(h5url, {path:"/var"});
            let _index = _var.attrs["_index"];
            varList = await dataUtils._hdf5Api.get(h5url, {path:"/var/" + _index});
        }
    }
    catch {
        varList = [];
    }
    return varList.value;
}

async function getObsList(h5url) {
    let obsList;
    obsList = await dataUtils._hdf5Api.getKeysNames(h5url, "/obs/");
    obsList = obsList.filter(obs => obs !== "_index");
    return obsList;
}

function csrToCsc(csrData) {
    const numRows = csrData.numRows;
    const numCols = csrData.numCols;
    const csrValues = csrData.values;
    const csrIndices = csrData.indices;
    const csrIndptr = csrData.indptr;

    const numNonZero = csrValues.length;
    const cscValues = new Float64Array(numNonZero);
    const cscIndices = new Int32Array(numNonZero);
    const cscIndptr = new Int32Array(numCols + 1).fill(0);

    for (let i = 0; i < numNonZero; i++) {
        cscIndptr[csrIndices[i] + 1]++;
    }

    for (let i = 0, sum = 0; i < numCols; i++) {
        const temp = cscIndptr[i];
        cscIndptr[i] = sum;
        sum += temp;
    }
    cscIndptr[numCols] = numNonZero;

    for (let i = 0; i < numRows; i++) {
        for (let j = csrIndptr[i], jEnd = csrIndptr[i + 1]; j < jEnd; j++) {
            const col = csrIndices[j];
            const dst = cscIndptr[col];

            cscValues[dst] = csrValues[j];
            cscIndices[dst] = i;
            cscIndptr[col]++;
        }
    }

    /*for (let i = numCols - 1, sum = 0; i >= 0; i--) {
        const temp = cscIndptr[i];
        cscIndptr[i] = sum;
        sum += temp;
    }*/
    for (let i = numCols - 1, sum = 0; i >= 0; i--) {
        const temp = cscIndptr[i];
        cscIndptr[i] = sum;
        sum += temp;
    }

    return {
        numRows: numCols,
        numCols: numRows,
        values: cscValues,
        indices: cscIndices,
        indptr: cscIndptr
    };
}

h5Utils.h5ad_to_tmap = async function(h5url) {
    var img_key = "hires";
    var markerScale = 2.4;
    var plugins = [];

    var globalX = "", globalY = "";
    let loadingModal = interfaceUtils.loadingModal("Loading data, please wait...");
    try {
        var adataCoord = await dataUtils._hdf5Api.getKeysNames(h5url, "/obsm/");
    }
    catch {
        if (window[h5url] === undefined) {
            let oldName = h5url;
            window[h5url] = await dataUtils.relocateOnDisk(h5url)
            h5url = window[h5url];
            if (!(typeof h5url === 'string' || h5url instanceof String))
                window[h5url.name] = h5url
            return await h5Utils.h5ad_to_tmap (h5url);
        }
        else if (window[h5url] == h5url) {
            $(loadingModal).modal("hide");
            interfaceUtils.alert ("Impossible to load " + h5url);
            return;
        }
        else {
            h5url = window[h5url];
            return h5Utils.h5ad_to_tmap (h5url);
        }
    }
    var coordinates_list = ["spatial", "X_spatial", "X_umap", "tSNE"];
    for (var i = 0; i < coordinates_list.length; i++) {
        var coordinates = coordinates_list[i];
        if (adataCoord.includes(coordinates)) {
            globalX = "/obsm/" + coordinates + ";0";
            globalY = "/obsm/" + coordinates + ";1";
            break;
        }
    }

    var layers = [];
    var library_ids = await dataUtils._hdf5Api.getKeysNames(h5url, "/uns/spatial");
    /*try {
        library_ids = await dataUtils._hdf5Api.getKeysNames(h5url, "/obs/library_id/categories");
        console.log(library_ids);

    } catch (error) {
        // do nothing
    }*/

    var coord_factor = 1;
    for (var i = 0; i < library_ids.length; i++) {
        var library_id = library_ids[i];
        coord_factor = await dataUtils._hdf5Api.get(h5url, {path:"/uns/spatial/" + library_id + "/scalefactors/tissue_" + img_key + "_scalef"});
        coord_factor = coord_factor.value;
        layers.push({
            "name": library_id,
            "tileSource": (h5url.name || h5url) +
                "?h5path=" +
                "/uns/spatial/" +
                library_id +
                "/images/" +
                img_key,
        });
    }

    var use_libraries = library_ids.length > 1;

    var library_col = "";
    if (use_libraries) {
        // TODO
    }

    var spatial_connectivities = "";
    let obsp = await dataUtils._hdf5Api.getKeysNames(h5url, "/obsp/");
    if ((obsp).includes("spatial_connectivities")) {
        spatial_connectivities = "/obsp/spatial_connectivities;join";
    }

    /*var encodingType = null;
    if ("encoding-type" in (await dataUtils._hdf5Api.getKeysNames(h5url, "/X")).attrs.keys()) {
        encodingType = "encoding-type";
    } else if ("h5sparse_format" in (await dataUtils._hdf5Api.getKeysNames(h5url, "/X")).attrs.keys()) {
        encodingType = "h5sparse_format";
    }
    if (encodingType) {
        if ((await dataUtils._hdf5Api.getKeysNames(h5url, "/X")).attrs[encodingType] == "csr_matrix") {
            // TODO to_csc_sparse(adata);
        }
    }*/
    
    var varList = await getVarList(h5url);
    var obsList = await getObsList(h5url);

    var new_tmap_project = JSON.parse(JSON.stringify(tmap_template));
    var unsList = await dataUtils._hdf5Api.getKeysNames(h5url, "/uns/");
    new_tmap_project["layers"] = layers;
    new_tmap_project["plugins"] = plugins;
    /*if ("tmap" in unsList) {
        new_tmap_project = JSON.parse(
            adata.get(
                "/uns/tmap",
                "{}",
            )
        );
        if (!("markerFiles" in new_tmap_project.keys())) {
            new_tmap_project["markerFiles"] = [];
        }
    }*/
    obsListCategorical = [];
    obsListNumerical = [];
    palette = {};
    for (obs of obsList) {
        let keys = await dataUtils._hdf5Api.getKeysNames(h5url, "/obs/" + obs);
        if (keys.includes("categories")) {
            obsListCategorical.push(obs);
            let p = await getPalette(h5url, obs);
            palette[obs] = JSON.stringify(p);
        }
        else {
            obsListNumerical.push(obs);
        }
    }
    new_tmap_project["markerFiles"].push(
        {
          "expectedHeader": {
            "X": globalX,
            "Y": globalY,
            "cb_cmap": "interpolateTurbo",
            "cb_col": "",
            "cb_gr_dict": "",
            "gb_col": "",
            "gb_name": "",
            "opacity": "1",
            "pie_col": "",
            "scale_col": "",
            "scale_factor": markerScale,
            "coord_factor": coord_factor,
            "shape_col": "",
            "shape_fixed": "disc",
            "shape_gr_dict": "",
            "edges_col": spatial_connectivities,
            "collectionItem_col": library_col,
            "collectionItem_fixed": "0",
          },
          "expectedRadios": {
            "cb_col": true,
            "cb_gr": false,
            "cb_gr_dict": true,
            "cb_gr_key": true,
            "cb_gr_rand": false,
            "pie_check": false,
            "scale_check": false,
            "sortby_check": true,
            "shape_col": false,
            "shape_fixed": true,
            "shape_gr": false,
            "shape_gr_dict": false,
            "shape_gr_rand": true,
            "collectionItem_col": use_libraries,
            "collectionItem_fixed": !use_libraries,
          },
          "hideSettings": true,
          "name": "Numerical observations",
          "path": h5url.name || h5url,
          "dropdownOptions": obsListNumerical
            .map((obs) => {
              return {
                optionName: obs,
                name: obs,
                "expectedHeader.cb_col": `/obs/${obs}`,
                "expectedHeader.sortby_col": `/obs/${obs}`,
              };
            }),
          "title": "Numerical observations",
          "uid": "mainTab",
        }
    );new_tmap_project.markerFiles.push({
        expectedHeader: {
            X: globalX,
            Y: globalY,
            gb_col: "obs",
            opacity: "1",
            scale_factor: markerScale,
            coord_factor: coord_factor,
            shape_fixed: "disc",
            edges_col: spatial_connectivities,
            collectionItem_col: library_col,
            collectionItem_fixed: "0"
        },
        expectedRadios: {
            cb_col: false,
            cb_gr: true,
            cb_gr_dict: true,
            cb_gr_key: false,
            cb_gr_rand: false,
            pie_check: false,
            scale_check: false,
            shape_col: false,
            shape_fixed: true,
            sortby_check: false,
            shape_gr: false,
            shape_gr_dict: false,
            shape_gr_rand: true,
            collectionItem_col: use_libraries,
            collectionItem_fixed: !use_libraries
        },
        hideSettings: true,
        name: "Categorical observations",
        path: h5url.name || h5url,
        dropdownOptions: obsListCategorical.map(obs => ({
            optionName: obs,
            name: obs,
            "expectedHeader.gb_col": `/obs/${obs}`,
            "expectedHeader.cb_gr_dict": palette[obs]
        })),
        title: "Categorical observations",
        uid: "mainTab"
    });
    
    new_tmap_project.markerFiles.push({
        expectedHeader: {
            X: globalX,
            Y: globalY,
            cb_cmap: "interpolateViridis",
            cb_col: "",
            scale_factor: markerScale,
            coord_factor: coord_factor,
            shape_fixed: "disc",
            edges_col: spatial_connectivities,
            collectionItem_col: library_col,
            collectionItem_fixed: "0"
        },
        expectedRadios: {
            cb_col: true,
            cb_gr: false,
            cb_gr_dict: false,
            cb_gr_key: true,
            cb_gr_rand: false,
            pie_check: false,
            scale_check: false,
            shape_col: false,
            shape_fixed: true,
            sortby_check: true,
            shape_gr: false,
            shape_gr_dict: false,
            shape_gr_rand: true,
            collectionItem_col: use_libraries,
            collectionItem_fixed: !use_libraries
        },
        hideSettings: true,
        name: "Gene expression",
        path: h5url.name || h5url,
        dropdownOptions: varList.map((gene, index) => ({
            optionName: gene,
            name: "Gene expression: " + gene,
            "expectedHeader.cb_col": `/X;${index}`,
            "expectedHeader.sortby_col": `/X;${index}`
        })),
        title: "Gene expression",
        uid: "mainTab"
    });
    
    projectUtils.loadProject(new_tmap_project);
    $(loadingModal).modal("hide");
}


