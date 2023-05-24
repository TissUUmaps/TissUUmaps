/*import * as hdf5 from "../../vendor/h5wasm/hdf5_hl.js";
import  { createLazyFile } from '../../vendor/h5wasm/lazyFileLRU.js';*/
importScripts("../../vendor/h5wasm/h5wasm_iife.js");
importScripts("../../vendor/h5wasm/lazyFileLRU.js");

var file = {};

function getAttributes (item) {
    var attrs = {};
    for (let key in item.attrs) {
        attrs[key] = item.attrs[key].value;
    }
    return attrs;
}

self.onmessage = async function (event) {
    const { action, payload, id } = event.data;
    if (action === "load") {
        var url = payload?.url;
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
                createLazyFile(FS, '/', 'current_file_with_range'+id+'.h5', true, false, config);
                file[url] = new h5wasm.File('current_file_with_range'+id+'.h5');
            } catch (error) {
                h5wasm.FS.createLazyFile('/', "current_file_without_range"+id+".h5", url, true, false);
                file[url] = new h5wasm.File("current_file_without_range"+id+".h5");
            }
        }
        else {
            FS.mkdir('/work');
            FS.mount(FS.filesystems.WORKERFS, { files: [url] }, '/work');

            file[url.name] = new h5wasm.File(`/work/${url.name}`, 'r');
            url = url.name
        }
        self.postMessage({id:id,data:file[url].keys()})
    }
    else if (action === "keys") {
        await h5wasm.ready;
        const url = payload?.url;
        if (file[url]) {
            const path = payload?.path ?? "/";
            const item = file[url].get(path);
            if (item instanceof h5wasm.Group) {
                self.postMessage({
                    id:id,
                    type: item.type,
                    attrs: getAttributes (item),
                    children: [...item.keys()].map((x)=>{return path+"/"+x})
                });
            }
            else if (item instanceof h5wasm.Dataset) {
                self.postMessage({
                    id:id,
                    type: item.type,
                    shape: item.shape,
                    dtype: item.dtype,
                    attrs: getAttributes (item)
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
        const url = payload?.url;
        if (file[url]) {
            const path = payload?.path ?? "/";
            const item = file[url].get(path);
            self.postMessage({
                    id:id,
                    type: item.type,
                    attrs: getAttributes (item)
            });
        }
    }
    else if (action === "get") {
        await h5wasm.ready;
        const url = payload?.url;
        if (file[url]) {
            const path = payload?.path ?? "/";
            const item = file[url].get(path);
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
    else if (action === "clear") {
        file = {};
    }
  };