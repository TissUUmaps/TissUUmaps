/**
 * @file jnotebook.js
 * @author Christophe Avenel
 */

/**
 * @namespace jnotebook
 * @classdesc The root namespace for jnotebook.
 */
var jnotebook;
jnotebook = {
    functions:[
        /*{
            name:"",
            function:"loadjnotebook"
        }*/
    ]
 }

/**
 * This method is called when the document is loaded. The tmapp object is built as an "app" and init is its main function.
 * Creates the OpenSeadragon (OSD) viewer and adds the handlers for interaction.
 * To know which data one is referring to, there are Object Prefixes (op). For In situ sequencing projects it can be "ISS" for
 * Cell Profiler data it can be "CP".
 * If there are images to be displayed on top of the main image, they are stored in the layers object and, if there are layers
 * it will create the buttons to display them in the settings panel.
 * The SVG overlays for the viewer are also initialized here 
 * @summary After setting up the tmapp object, initialize it*/
jnotebook.init = function (tmappObject) {
    jnotebook.tmapp = tmappObject;
    jnotebook.functions.forEach(function(funElement, i) {
        var aElement = document.createElement("a");
        aElement.href = "#";
        aElement.addEventListener("click",function (event) {
            console.log("Click", event, funElement.function);
            window["jnotebook"][funElement.function]();
        });
        var spanElement = document.createElement("span");
        aElement.appendChild(spanElement);
        spanElement.innerHTML = funElement.name;
        dropdownMenu = document.getElementById("dropdown-menu-jnotebook");
        dropdownMenu.appendChild(aElement);
    });
    $("#ISS_collapse_btn").click();
    $(".navbar-default").hide();
    $(".ISS_viewer").css("height","100vh");
    
    function receiveMessage(event)
    {
        console.log("Message received", event);
        if (event.origin !== "http://localhost:8888")
            return;
        
        jnotebook.receiveMessage (event);
    }
    window.addEventListener("message", receiveMessage, false);
}

jnotebook.receiveMessage = function (event) {
    console.log("Message received", event)
}