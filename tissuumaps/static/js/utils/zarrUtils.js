// Import invidual classes and functions.
import { HTTPStore, openArray, openGroup, slice, containsGroup, containsArray , normalizeStoreArgument } from "../../vendor/zarr/zarr.mjs";

// Or import everything in one go
// import * as zarr from "https://cdn.skypack.dev/zarr";
/*
const z = await openArray ({
    store: "http://localhost:5000/merfish/data.zarr",
    path: "table/table/X",
    mode: "r"
});
console.log(z);
const arr1 = await z.get([slice(null,1), null]);
console.log(arr1);

const z2 = await openArray ({
    store: "http://localhost:5000/merfish/data.zarr",
    path: ".zmetadata",
    mode: "r"
});
console.log(z2);
const arr2 = await z2.get([0]);
console.log(arr2);
*/


/**
* @file zarrUtils.js Utilities for zarr-based marker loading
* @author Christophe Avenel
* @see {@link zarrUtils}
*/

/**
 * @namespace zarrUtils
 * @property {Boolean} _initialized True when zarrUtils has been initialized
 *
zarrUtils = {
    worker_path: 'js/utils/zarrUtils_worker.js',
    relative_root: '../../'
 }*/

class Zarr_API {
    constructor() {
    }
  
    async get (url, payload, action) {
        if (action === undefined) action = "get"; 
        let store = normalizeStoreArgument(url);
        const isGroup = await containsGroup (store, payload.path);
        let zObject = null;
        if (isGroup) {
            zObject = await openGroup (
                store,
                payload.path,
                "r",
                undefined,
                false
            );
        }
        else {
            const isArray = await containsArray (store, payload.path);
            if (isArray) {
                zObject = await openArray ({
                    store: store,
                    path: payload.path,
                    mode: "r",
                    cacheAttrs: false
                });
            }
        }
        zObject.attrs = await zObject.attrs.asObject();
        if (action == "attr") {
            console.log(zObject);
            //zObject.attrs = await zObject.attrs.asObject();
            //zObject.attrs.shape = zObject.meta.shape;
            return zObject;
        }
        else if (action == "get") {
            console.log(payload);
            console.log(zObject);
            if (!payload.slice) {
                payload.slice = [null]
            }
            else {
                payload.slice = payload.slice.map((x) => {
                    console.log(x);
                    if (x.length == 0) return null;return slice(...x)
                })
            }
            console.log(payload.slice);
            var outputArray = await zObject.get(payload.slice);
            outputArray.value = outputArray.data;
            return outputArray
        }
        else if (action == "keys") {
            console.log("Keys not yet supported!");
        }
    }
}

class NGFF_API  extends Zarr_API {
        
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
                console.log("getXRow", data_X);
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
                console.log(data_X);
                console.log("data_X.attrs[encoding-type]", data_X.attrs["encoding-type"])
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
                    interfaceUtils.alert("csr sparse format not supported.")
                    resolve("csr sparse format not supported!")
                }
                else {
                    return this.getXRow_array (url, rowIndex, path).then((data)=>{
                        resolve(data);
                    });
                }
            });
        });
    }
    
    getKeys (url, path) {
        if (path === undefined) path = "/";
        if (path[0] != "/") path = "/" + path;
        return new Promise(resolve => {
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
                else {
                    path = path.substring(0, Math.max(path.lastIndexOf('/'),path.lastIndexOf(';')));
                    this.getKeys(url, path).then((data_keys_root)=>{
                        resolve(data_keys_root);
                    })
                }
            });
        });
    }
}
/*
var hdf5Api = new Zarr_API()
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

dataUtils._zarrApi = new NGFF_API();