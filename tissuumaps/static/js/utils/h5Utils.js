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
    worker_path: 'js/utils/h5Utils_worker.js',
    relative_root: '../../'
 }

class H5_API {
    constructor() {
        this.chunkSize = 1024 * 1024;
        this.resolvers = {};
        this.count = 0; // used later to generate unique ids
        this.status = {}

        this.worker = new Worker(h5Utils.worker_path);
        this.worker.addEventListener('message', (e) => {
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
                _url = h5Utils.relative_root + _url;
        this.worker.postMessage({id: id, action: "load", payload: {requestChunkSize, url:_url}});
        return new Promise(resolve => this.resolvers[id] = resolve);
    }

    load (url) {
        if (typeof url === 'string' || url instanceof String) {
            var urlName = url;
        }
        else {
            var urlName = url.name;
        }
        this.status[urlName] = "loading";
        this.loadPromise(url).then((data)=>{
            setTimeout(()=>{this.status[urlName] = "loaded";},50);
        })
    }

    get (url, payload, action) {
        if (action === undefined) action = "get"; 
        function sleep (time) {
            return new Promise((resolve) => setTimeout(resolve, time));
        }
        if (!(typeof url === 'string' || url instanceof String)) {
            var urlName = url.name;
        }
        else {
            var urlName = url;
        }
        console.log("get:",url, urlName, this.status[urlName]);
        if (this.status[urlName] === undefined) {
            this.load(url);
        }
        return new Promise(resolve => {
            if (this.status[urlName] === "loading") {
                sleep(50).then (()=>{
                    this.get(url, payload, action).then((data)=>{
                        resolve(data)
                    })
                });
                return;
            }
            const id = this.count++;
            this.resolvers[id] = resolve
            if (typeof url === 'string' || url instanceof String) {
                if (!url.startsWith("https")) {
                    payload.url = h5Utils.relative_root + url;
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
