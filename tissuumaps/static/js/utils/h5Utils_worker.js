/*import * as hdf5 from "../../vendor/h5wasm/hdf5_hl.js";
import  { createLazyFile } from '../../vendor/h5wasm/lazyFileLRU.js';*/
importScripts("../../vendor/h5wasm/h5wasm_iife.js");
importScripts("../../vendor/h5wasm/lazyFileLRU.js");

var file;

function getAttributes (item) {
    var attrs = {};
    for (let key in item.attrs) {
        attrs[key] = item.attrs[key].value;
    }
    return attrs;
}

function getLeafNodes(nodes, result = []){
    for(var i in nodes.children){
        if(Object.keys(nodes.children[i].children).length == 0){
            result.push(nodes.children[i].value);
        }else{
            result = getLeafNodes(nodes.children[i], result);
        }
    }
    return result;
}

function getKeys (path) {
    const item = file.get(path);
    if (path == "/") path = "";
    let attributes = getAttributes (item);
    var keys_array = {
        children:{},
        attrs:attributes,
        value:path
    }
    var stop_encoding = [
        "csc_matrix",
        "csr_matrix",
        "array",
        "categorical"
    ]
    if (stop_encoding.indexOf(attributes["encoding-type"]) >= 0) return keys_array;
    if (!item) return keys_array;
    if (!item.keys) return keys_array;
    for (let key of item.keys()) {
        keys_array.children[key] = getKeys(path + "/" + key)
    }
    return keys_array;
}

self.onmessage = async function (event) {   
    const { action, payload, id } = event.data;
    if (action === "load") {
        const url = payload?.url;
        console.log("url", url);
        const { FS } = await h5wasm.ready;
        if (typeof url === 'string' || url instanceof String) {
            const requestChunkSize = payload?.requestChunkSize ?? 1024 * 1024;
            const LRUSize = payload?.LRUSize ?? 50;
            const config = {
                rangeMapper: (fromByte, toByte) => ({url, fromByte, toByte}),
                requestChunkSize,
                LRUSize
            }
            try {
                createLazyFile(FS, '/', 'current_file_with_range.h5', true, false, config);
                file = new h5wasm.File("current_file_with_range.h5");
            } catch (error) {
                h5wasm.FS.createLazyFile('/', "current_file_without_range.h5", url, true, false);
                file = new h5wasm.File("current_file_without_range.h5");
            }
        }
        else {
            FS.mkdir('/work');
            FS.mount(FS.filesystems.WORKERFS, { files: [url] }, '/work');

            file = new h5wasm.File(`/work/${url.name}`, 'r');
        }
        self.postMessage({id:id,data:file.keys()})
    }
    else if (action === "keys") {
        await h5wasm.ready;
        if (file) {
            const path = payload?.path ?? "/";
            /*var keys = getKeys(path);
            console.log(keys);
            var leafs = getLeafNodes(keys);
            console.log(leafs)
            self.postMessage({
                id:id,
                keys: leafs
            });*/
            const item = file.get(path);
            if (item instanceof h5wasm.Group) {
                self.postMessage({
                    id:id,
                    type: item.type,
                    attrs: getAttributes (item),
                    children: [...item.keys()].map((x)=>{return path+"/"+x})
                });
            }
            else {
                self.postMessage({
                    id:id,
                    type: "error",
                    value: `item ${path} not found or is not a group`
                })
            }
        }
    }
    else if (action === "attr") {
        await h5wasm.ready;
        if (file) {
            const path = payload?.path ?? "/";
            const item = file.get(path);
            self.postMessage({
                    id:id,
                    type: item.type,
                    attrs: getAttributes (item)
            });
        }
    }
    else if (action === "get") {
        await h5wasm.ready;
        if (file) {
            const path = payload?.path ?? "/";
            const item = file.get(path);
            if (item instanceof h5wasm.Group) {
                self.postMessage({
                    id:id,
                    type: item.type,
                    attrs: getAttributes (item),
                    children: [...item.keys()] 
                });
            } else if (item instanceof h5wasm.Dataset) {
                const value = (payload.slice) ? item.slice(payload.slice) : item.value;
                self.postMessage({
                    id:id,
                    type: item.type,
                    attrs: getAttributes (item),
                    value
                });
            } else if (item instanceof h5wasm.BrokenSoftLink || item instanceof h5wasm.ExternalLink) {
                self.postMessage({id:id, item: item});
            }
            else {
                self.postMessage({
                    id:id,
                    type: "error",
                    value: `item ${path} not found`
                })
            }
        }
    }
  };