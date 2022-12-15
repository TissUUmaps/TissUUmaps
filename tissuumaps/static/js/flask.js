flask = {}

flask.init = function () {
    $(document).on( "click", ".layerSettingButton", function(){
        const queryString = window.location.search;
        const urlParams = new URLSearchParams(queryString);
        var path = urlParams.get('path');
        if (path == null) {
            path = "";
        }
        interfaceUtils.alert(
            `
            <iframe src='${path}/${$(this).data('source')}/info' style="width:100%;min-height:500px;"></iframe>
            `, "Image information"
        )
    } );
    $("head").append('<style id="flask-css" type="text/css"></style>');
    var sheet = $("#flask-css").get(0).sheet;
    var styles = ".layerSettingButton:after {";
    styles += "content: '\\f431';";
    styles += "font-family: 'bootstrap-icons';";
    styles += "margin: 0px 0px 0px 5px;";
    styles += "}";
    // Add the first CSS rule to the stylesheet
    sheet.insertRule(styles, 0);
}

flask.standalone = {}

flask.standalone.init = function () {
    flask.init();
    flask.standalone.backend = null;
    new QWebChannel(qt.webChannelTransport, function (channel) {
        flask.standalone.backend = channel.objects.backend;
    });

    // Add layer button
    div = HTMLElementUtils.createElement({"kind":"div", extraAttributes:{"class":"px-3 my-2"}});
    button = HTMLElementUtils.createElement({"kind":"div", extraAttributes:{"class":"btn btn-primary btn-sm"}});
    button.innerHTML = "Add image layer";
    div.append(button)
    document.getElementById("image-overlay-panel").after(div)
    button.addEventListener("click", function(){
        flask.standalone.addLayer("");
    });
    flask.standalone.pixelFlickering = HTMLElementUtils.createElement({"kind":"div", extraAttributes:{"style":"position:absolute;right:0px;bottom:0px;width:1px;height:1px;line-height:1px;background-color:#FFFFFF"}});
    document.body.append(flask.standalone.pixelFlickering)
    setInterval(
        function() {
            if (flask.standalone.pixelFlickering.style.backgroundColor=="rgb(255, 255, 255)")
                flask.standalone.pixelFlickering.style.backgroundColor="rgb(255, 255, 254)";
            else 
                flask.standalone.pixelFlickering.style.backgroundColor="rgb(255, 255, 255)";
        },
        50
    )
    var plusButton = document.getElementById("plus-1-button");
    var clone = plusButton.cloneNode(true); plusButton.replaceWith(clone); 
    document.getElementById("plus-1-button").addEventListener("click", function(event) {
        event.stopPropagation();
        const queryString = window.location.search;
        const urlParams = new URLSearchParams(queryString);
        const path = urlParams.get('path')
        flask.standalone.backend.addCSV(path, "", function(csvJSON) {
            if (csvJSON["markerFile"]!=null) {
                interfaceUtils.generateDataTabUI(csvJSON["markerFile"]);
            }
        });
        return false;
    }, true);
};

flask.standalone.addCSV = function (filename) {
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const path = urlParams.get('path')
    flask.standalone.backend.addCSV(path, filename, function(csvJSON) {
        if (csvJSON["markerFile"]!=null) {
            interfaceUtils.generateDataTabUI(csvJSON["markerFile"]);
        }
    });
}

flask.standalone.addLayer = function (filename) {
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const path = urlParams.get('path')
    flask.standalone.backend.addLayer(path, filename, function(layerImg) {
        if (layerImg["dzi"]!=null) {
            var layerName = layerImg["name"];
            var tileSource = layerImg["dzi"];
            tmapp.layers.push({
                name: layerName,
                tileSource: tileSource
            });
            var i = tmapp.layers.length - 2;
            var layer = {"name":layerName, "tileSource":tileSource};
            overlayUtils.addLayer(layer, i, true);
            overlayUtils.addLayerSettings(layerName, tileSource, i, true);
        }
    });
}

flask.standalone.saveProject = function () {
    projectUtils.getActiveProject().then((state) => {
        setTimeout(function() {
            flask.standalone.backend.saveProject(JSON.stringify(state))
        }, 300);
    });
};

flask.standalone.exportToStatic = function () {
    projectUtils.getActiveProject().then((state) => {
        var loadingModal =null;
        setTimeout(function() {
            loadingModal=interfaceUtils.loadingModal("Exporting to static web page")
        },0);
        setTimeout(function() {
            flask.standalone.backend.exportToStatic(JSON.stringify(state),
                function (jsonVal) {
                    $(loadingModal).modal('hide');
                    if (jsonVal["success"]) {
                        interfaceUtils.alert("Exporting done.")
                    }
                    else {
                        interfaceUtils.alert("Exporting failed: " + jsonVal["error"])
                    }
                }
            );
        }, 500);
    })
};

flask.server = {}

flask.server.init = function () {
    flask.init();
    document.getElementById("menubar_File_Import").classList.add("d-none");
    document.getElementById("menubar_File_Export").classList.add("d-none");

    interfaceUtils.addMenuItem(["File","Save project"],function(){
        var modalUID = "messagebox";
        projectUtils.getActiveProject().then((state) => {
            interfaceUtils.prompt("Save project under the name:","NewProject","Save project")
            .then((filename) => {
                state.filename = filename;
                if (filename.split('.').pop() != "tmap") {
                    filename = filename + ".tmap"
                }
                const queryString = window.location.search;
                const urlParams = new URLSearchParams(queryString);
                const path = urlParams.get('path')
                $.ajax({
                    type: "POST",
                    url: "/" + filename + "?path=" + path,
                    // The key needs to match your method's input parameter (case-sensitive).
                    data: JSON.stringify(state),
                    contentType: "application/json; charset=utf-8",
                    dataType: "json",
                    success: function(data) {
                        $('#loadingModal').modal('hide');
                    },
                    failure: function(errMsg) {
                        $('#loadingModal').modal('hide');
                        alert(errMsg);
                    }
                });
            })
        })
    },true);


    interfaceUtils.addMenuItem(["File","Open"],function(){
        var modalUID = "messagebox"
        button1=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-primary mx-2"}})
        button1.innerText = "Cancel";
        button1.addEventListener("click",function(event) {
            $(`#${modalUID}_modal`).modal('hide');
        })
        buttons=divpane=HTMLElementUtils.createElement({"kind":"div"});
        buttons.appendChild(button1);
        content=HTMLElementUtils.createElement({"kind":"div"});
        content.innerHTML = "<iframe src='/filetree' width='100%' height='300px'></iframe>";
        interfaceUtils.generateModal ("Open file", content, buttons, modalUID);
    },true);

    // Add layer button
    div = HTMLElementUtils.createElement({"kind":"div", extraAttributes:{"class":"px-3 my-2"}});
    button = HTMLElementUtils.createElement({"kind":"div", extraAttributes:{"class":"btn btn-primary btn-sm"}});
    button.innerHTML = "Add image layer";
    div.append(button)
    document.getElementById("image-overlay-panel").after(div)
    button.addEventListener("click", function(){
        var modalUID = "messagebox"
        button1=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-primary mx-2"}})
        button1.innerText = "Cancel";
        button1.addEventListener("click",function(event) {
            $(`#${modalUID}_modal`).modal('hide');
        })
        buttons=divpane=HTMLElementUtils.createElement({"kind":"div"});
        buttons.appendChild(button1);
        content=HTMLElementUtils.createElement({"kind":"div"});
        content.innerHTML = "<iframe src='/filetree?addlayer=1' width='100%' height='300px'></iframe>";
        interfaceUtils.generateModal ("Open file", content, buttons, modalUID);
    });
}

flask.server.addLayer = function (filename) {
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const path = urlParams.get('path')
    if (filename.startsWith(path)) {
        filename = filename.slice(path.length);
    }
    else {
        interfaceUtils.alert("All layers must be in the same folder");
        return;
    }
    var layerName = filename.split('/').reverse()[0];;
    var tileSource = filename;
    tmapp.layers.push({
        name: layerName,
        tileSource: tileSource
    });
    var i = tmapp.layers.length - 2;
    var layer = {"name":layerName, "tileSource":tileSource};
    overlayUtils.addLayer(layer, i, true);
    overlayUtils.addLayerSettings(layerName, tileSource, i, true);
    var modalUID = "messagebox";
    $(`#${modalUID}_modal`).modal('hide');
}

flask.server.loading = function (message) {
    var loadingModal =null;
    setTimeout(function() {
        loadingModal=interfaceUtils.loadingModal(message);
    },0);
}

flask.server.hideloading = function (message) {
    $('#loadingModal').modal('hide');
}

function toggleNavbar(turn_on = null) {
    return false;
}


/**
 * Save the current canvas as a PNG image
 */

// Child website:
window.addEventListener("message", evt => {
    console.log("evt.data",evt.data) // "Question!"
    // TODO: use overlayUtils.getCanvasPNG as a promise to get png image
    
    // Create an empty canvas element
    var canvas = document.createElement("canvas");
    var ctx_osd = document.querySelector(".openseadragon-canvas canvas").getContext("2d");
    var ctx_webgl = document.querySelector("#gl_canvas").getContext("webgl2", glUtils._options);
    canvas.width = Math.min(ctx_osd.canvas.width, ctx_webgl.canvas.width);
    canvas.height = Math.min(ctx_osd.canvas.height, ctx_webgl.canvas.height);
    
    // Copy the image contents to the canvas
    var ctx = canvas.getContext("2d");
    
    ctx.drawImage(ctx_osd.canvas, 0, 0, canvas.width, canvas.height);
    ctx.drawImage(ctx_webgl.canvas, 0, 0, canvas.width, canvas.height);
    var dataURL = canvas.toDataURL("image/png");
    
    var svgString = new XMLSerializer().serializeToString(document.querySelector('.openseadragon-canvas svg'));

    var DOMURL = self.URL || self.webkitURL || self;
    var img = new Image();
    var svg = new Blob([svgString], {type: "image/svg+xml;charset=utf-8"});
    var url = DOMURL.createObjectURL(svg);
    img.onload = function() {
        ctx.drawImage(img, 0, 0);
        var png = canvas.toDataURL("image/png");
           
        evt.source.postMessage({"img":png,"type":"screenshot" }, evt.origin);
        DOMURL.revokeObjectURL(png);
    };
    img.src = url;

    
});
