import { ready } from "./hdf5_hl.js";
export const UPLOADED_FILES = [];
export async function uploader(event) {
    const { FS } = await ready;
    const target = event.target;
    let file = target.files?.[0]; // only one file allowed
    if (file) {
        let datafilename = file.name;
        let ab = await file.arrayBuffer();
        FS.writeFile(datafilename, new Uint8Array(ab));
        if (!UPLOADED_FILES.includes(datafilename)) {
            UPLOADED_FILES.push(datafilename);
            console.log("file loaded:", datafilename);
        }
        else {
            console.log("file updated: ", datafilename);
        }
        target.value = "";
    }
}
function create_downloader() {
    let a = document.createElement("a");
    document.body.appendChild(a);
    a.style.display = "none";
    a.id = "savedata";
    return function (data, fileName) {
        let blob = (data instanceof Blob) ? data : new Blob([data], { type: 'application/x-hdf5' });
        // IE 10 / 11
        const nav = window.navigator;
        if (nav.msSaveOrOpenBlob) {
            nav.msSaveOrOpenBlob(blob, fileName);
        }
        else {
            let url = window.URL.createObjectURL(blob);
            a.href = url;
            a.download = fileName;
            a.target = "_blank";
            //window.open(url, '_blank', fileName);
            a.click();
            setTimeout(function () { window.URL.revokeObjectURL(url); }, 1000);
        }
        // cleanup: this seems to break things!
        //document.body.removeChild(a);
    };
}
;
export const downloader = create_downloader();
export async function to_blob(hdf5_file) {
    const { FS } = await ready;
    hdf5_file.flush();
    return new Blob([FS.readFile(hdf5_file.filename)], { type: 'application/x-hdf5' });
}
export async function download(hdf5_file) {
    let b = await to_blob(hdf5_file);
    downloader(b, hdf5_file.filename);
}
export function dirlisting(path, FS) {
    let node = FS.analyzePath(path).object;
    if (node && node.isFolder) {
        let files = Object.values(node.contents).filter(v => !(v.isFolder)).map(v => v.name);
        let subfolders = Object.values(node.contents).filter(v => (v.isFolder)).map(v => v.name);
        return { files, subfolders };
    }
    else {
        return {};
    }
}
