/**
* @file interfaceUtils.js adding and managing elements in the interface
* @author Leslie Solorzano
* @see {@link interfaceUtils}
*/
/**
* @namespace interfaceUtils
*/
interfaceUtils={}


/** 
* @param {String} domid The id of the element to listen to
* @param {String} event The event to listen for
* @param {String} handler Function to answer with
* @param {Bool} debug If true will print to console
* @summary Listen to an event of an element, if the element doesnçt exist get a warning. */
interfaceUtils.listen= function(domid,event,handler,debug){
    var dbg=debug || false;
    var elem= document.getElementById(domid);
    if(elem){
        elem.addEventListener(event, handler);
        if(dbg){
            console.log(domid,event,String(handler));
        }
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }
}


/** Get the region to be analyzed  */
interfaceUtils.analyzeRegionUI = function (callingbutton) {
    var op = tmapp["object_prefix"];

	if (!dataUtils.data["gene"][op + "_barcodeGarden"]) {
		interfaceUtils.alert("Load markers first");
		return;
    }

    var regionid = callingbutton[0].getAttribute("parentRegion");
    regionUtils.analyzeRegion(regionid);
}


/** Get the region to be filled  */
interfaceUtils.fillRegionUI = function (callingbutton) {
    var regionid = callingbutton[0].getAttribute("parentRegion");
    regionUtils.fillRegion(regionid);
}

/** Delete a RegionUI  */
interfaceUtils.deleteRegionUI = function(callingbutton) {
    var regionid = callingbutton[0].getAttribute("parentRegion");
    regionUtils.deleteRegion(regionid);
}

/**
 * @param {HTMLelement} callingbutton Button element containing parent region information
 * @summary Get the info of the region that has to be changed */
interfaceUtils.changeRegionUI = function (callingbutton) {
    var regionid = callingbutton[0].getAttribute("parentRegion");
    regionUtils.changeRegion(regionid);
}

/** 
* @param {String} domid The id of the select element
* @param {String[]} elemlist Array of strings containing elements to add to the select
* @summary Add options to a select element */
interfaceUtils.addElementsToSelect=function(domid,elemlist){
    var select= document.getElementById(domid);
    if(select){
        elemlist.forEach(element => {
            var opt = document.createElement("option");
            opt.value= element;
            opt.innerHTML = element;
            select.appendChild(opt);
        });
    }else{
        console.log("Select with id "+domid+" doesn't exist");
        return null;
    }
}

/** 
* @param {String} domid The id of the select element
* @param {Object[]} Array of objects containing elements to add to the select
* @summary Add options to a select element using Objects with the keys: "value* and "innerHTML" */
interfaceUtils.addObjectsToSelect=function(domid,objlist){
    var select= document.getElementById(domid);
    if(select){
        objlist.forEach(element => {
            var opt = document.createElement("option");
            opt.value= element.value;
            opt.innerHTML = element.innerHTML;
            select.appendChild(opt);
        });
    }else{
        console.log("Select with id "+domid+" doesn't exist");
        return null;
    }
}

interfaceUtils.addSingleElementToSelect=function(domid,element,options){
    if(!options) options={};
    var select= document.getElementById(domid);
    if(select){       
        var opt = document.createElement("option");
        if(options.id) opt.id="region_opt_"+element;
        opt.value= element;
        opt.innerHTML = element;
        select.appendChild(opt);        
    }else{
        console.log("Select with id "+domid+" doesn't exist");
        return null;
    }
}


/** 
* @param {String} domid The id of the select element
* @summary Erase all options in a select element */
interfaceUtils.cleanSelect=function(domid){
    var select= document.getElementById(domid);
    if(select){       
        select.innerHTML = "";
    }else{
        console.log("Select with id "+domid+" doesn't exist");
        return null;
    }
}

/** 
* @param {String} domid The id of the element
* @summary Make an element invisible */
interfaceUtils.makeInvisible=function(domid){
    var elem= document.getElementById(domid);
    if(elem){
        elem.style.visibility="hidden";
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }
}

/** 
* @param {String} domid The id of the element
* @summary Make an element visible */
interfaceUtils.makeVisible=function(domid){
    var elem= document.getElementById(domid);
    if(elem){
        elem.style.visibility="visible";
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }
}

/** 
* @param {String} domid The id of the element
* @summary Disable an element */
interfaceUtils.disableElement=function(domid){
    var elem= document.getElementById(domid);
    if(elem){
        elem.disabled="true";
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }
}


/** 
* @param {String} domid The id of the element
* @summary Enable an element */
interfaceUtils.enableElement=function(domid){
    var elem= document.getElementById(domid);
    if(elem){
        elem.removeAttribute("disabled");
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }
}

/** 
* @param {String} domid The id of the element
* @return {Bool | null}
* @summary Ask if an element is enabled */
interfaceUtils.isEnabled=function(domid){
    var elem= document.getElementById(domid);
    if(elem){
        if(elem.hasAttribute("disabled")){
            return false;
        }else{ return true; }
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }
}

/** 
* @param {String} domid The id of the element
* @return {Object | null} Object with a "key" and a "value"
* @summary Get the selected option in a sleect element */
interfaceUtils.getSelectedIndexValue=function(domid){
    var selector= document.getElementById(domid);
    if(selector){
        var obj={};
        obj.key = selector.options[selector.selectedIndex].value;
        obj.value =selector.options[selector.selectedIndex].innerHTML;
        return obj;
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }

}

/** 
* @param {String} classname The class of the elements
* @return {HTMLelements[] | null} array of HTMl elements
* @summary Call the main dom.getElementsByClassName function with a warning if no elements exist */
interfaceUtils.getElementsByClassName=function(classname){
    var elems= document.getElementsByClassName(classname);
    if(elems){
        return elems;
    }else{
        console.log("No elements of class "+classname+" doesn't exist");
        return null;
    }
}


/** 
* @param {String} classname The class of the elements
* @return {HTMLelements[] | null} array of HTMl elements
* @summary Call the main dom.getElementsByTagName function with a warning if no elements exist */
interfaceUtils.getElementsByTagName=function(tagname){
    var elems= document.getElementsByTagName(tagname);
    if(elems){
        return elems;
    }else{
        console.log("No elements of class "+classname+" doesn't exist");
        return null;
    }
}

/** 
* @param {String} domid The id of the element
* @param {String} choice thing to change
* @param {String} value to change it to
* @return {HTMLelement | null} HTMl element
* @summary Get the an element and warn if none exists */
interfaceUtils.setValueForElement=function(domid,choice, value){
    var elem= document.getElementById(domid);
    if(elem){
        elem[choice]=value
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }
}


/** 
* @param {String} domid The id of the element
* @param {String} attr thing to change
* @param {String} value to change it to
* @return {HTMLelement | null} HTMl element
* @summary Get the an element and warn if none exists */
interfaceUtils.setAttributeForElement=function(domid,attr, value){
    var elem= document.getElementById(domid);
    if(elem){
        elem.setAttribute(attr, value);
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }
}
/** 
* @param {String} domid The id of the element
* @param {Boolean} debug Print warnings if true
* @return {HTMLelement | null} HTMl element
* @summary Get the an element and warn if none exists */
interfaceUtils.getElementById=function(domid, debug=true){
    var elem= document.getElementById(domid);
    if(elem){
        return elem;
    }else{
        if (debug) console.log("Element with id "+domid+" doesn't exist");
        return null;
    }
}

/** 
* @param {String} domid The id of the element
* @return {String | null} HTMl element
* @summary Get the value of a dom element and warn if element does not exist*/
interfaceUtils.getValueFromDOM=function(domid){
    var elem= document.getElementById(domid);
    if(elem){
        return elem.value;
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }
}


/** 
* @param {String} domid The id of the element
* @return {String | null} innerHTMl
* @summary Get the innerHTML of a dom element and warn if element does not exist*/
interfaceUtils.getInnerHTMLFromDOM=function(domid){
    var elem= document.getElementById(domid);
    if(elem){
        return elem.innerHTML;
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }
}

interfaceUtils.removeOptionFromSelect=function(domid,key){
    var select= document.getElementById(domid);
    if(select){
        var remove=0;
        for (var i=0; i<select.length; i++){
            if (select.options[i].value == key )
                remove=i;
        }
        select.remove(remove);
    }else{
        console.log("Select with id "+domid+" doesn't exist");
        return null;
    }
}

interfaceUtils.emptyViewers = function (options) {
    var containers = options.containers || ["fixed", "moving"];
    containers.forEach(function (c) {
        var container = document.getElementById(c + "_viewer");
        while (container.lastChild) {
            container.removeChild(container.lastChild);
        }
    });  
}


/** 
* @param {String} dzi Path and name of the DZI file
* @param {String} viewer String that identifies a viewer and its 
* @summary associated components. For a single viewer the default is "ISS", 
* resulting in "ISS_viewer" as an identifier
* Open a DZI in a specific viewer. If a main location for images 
* is specified previously using the "url_prefix" variable, 
* it will be added to the dzi string */
interfaceUtils.openDZI=function(dzi,viewer){
    var possibleurlprefix=interfaceUtils.getValueFromDOM("url_prefix");
    if(possibleurlprefix){
        tmcpoints.url_prefix=possibleurlprefix;
    }
    tmcpoints[viewer+"_viewer"].open(tmcpoints.url_prefix + dzi);
    
}

/** 
* @param {String} domid The id of the element
* @summary See if a dom is checked (mostly a checkbox) */
interfaceUtils.isChecked=function(domid){
    var check= document.getElementById(domid);
    if(check){
        var checked=check.checked;
        return checked;
    }else{
        console.log("Check with id "+domid+" doesn't exist");
        return null;
    }
}

/** 
* @param {String} domid The id of the element
* @summary check if an input has o has not its first options sslected */
interfaceUtils.checkSelectNotZero=function(domid){
    var selector= document.getElementById(domid);
    if(selector){
        var key = selector.options[selector.selectedIndex].value;
        if(key.toString()=="0") 
            return false;
        return true;
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }

}


/** 
 * @param {object} ul dom object of the a tag 
 * @summary find and actiate main tabs */
interfaceUtils.activateMainChildTabs=function(elid){
    if (!document.getElementById(elid)) {
        return;
    }
    //first, find children ul and then their main children onwards
    children=document.getElementById(elid).getElementsByTagName("a");
    maintabids=[];
    nonmainids=[];
    //find the main a and its corresponding panel and activate it.
    for(var i=0;i<children.length;i++){
        if(children[i].classList.contains("main-child")){
            maintabids.push(children[i].href.split("#")[1]);
            children[i].classList.add("active")
        }else{
            nonmainids.push(children[i].href.split("#")[1]);
            children[i].classList.remove("active")
        }
    }

    //console.log(maintabids,nonmainids)
    
    for(var i=0;i<maintabids.length;i++){
        var elem=document.getElementById(maintabids[i]);
        if(elem)
            elem.classList.add("active")
        else{
            console.log("element "+maintabids[i]+" doesn't exist");
        }
    }

    for(var i=0;i<nonmainids.length;i++){
        var elem=document.getElementById(nonmainids[i]);
        if(elem)
            elem.classList.remove("active")
        else{
            console.log("element "+nonmainids[i]+" doesn't exist");
        }   
    }
}

/** 
 * @param {object} a dom object of the a tag 
 * @summary hides all the tabs that should not he  displayed except a itself */
interfaceUtils.hideTabsExcept = function (a) {
    //get a tag, get it's closes ul check the level, deactivate all but this
    const regex1 = RegExp("L([0-9]+)-tabs", 'g');
    //first, get closest ul contaninig list of a links
    var closestul = a.closest("ul");
    var level = 0;

    //find main child tabs and activate them
    
    //find href to know which id to look for and which to hide
    var elid = a[0].href.split("#")[1]
    interfaceUtils.activateMainChildTabs(elid);

    //check for this ul's classes to see if any matches regex
    if (closestul !== null) {
        closestul[0].classList.forEach(
            function (v) {
                var arr = regex1.exec(v)
                if (arr !== null)
                    level = Number(arr[1]);
            }
        )
        
    } else {
        console.log("no tabs for this a tag");
    }

    var findthislevel = "L" + String(level) + "-tabs";

    var uls = document.getElementsByClassName(findthislevel);

    //find all a tags in this levels and their hrefs
    var as = [];

    for (var i = 0; i < uls.length; i++) {
        var ulsas = uls[i].getElementsByTagName("a");
        for (var j = 0; j < ulsas.length; j++) {
            ana=ulsas[j].href.split("#")[1];
            if(!ana.includes(elid)){
                //only turn non elids
                as.push(ana)
                ulsas[j].classList.remove("active")
            }
        }    
    }

    for(var i=0;i<as.length;i++){
        //find elements with this id and deactivate them
        var el=document.getElementById(as[i]);
        
        if(el!==null && el.classList.length>0){
            el.classList.remove("active");
            el.classList.remove("show");
        }
    }
   
}

/** 
 * @param {object} a dom object of the a tag 
 * @summary hides all the tabs that should not he  displayed except a itself */
 interfaceUtils.toggleRightPanel = function (a) {
    var op = tmapp["object_prefix"];
    var menu=document.getElementById(op + "_menu");
    var main=document.getElementById(op + "_viewer_container");
    var btn=document.getElementById(op + "_collapse_btn");
    var style = window.getComputedStyle(menu);
    if (style.display === 'none') {
        menu.style.display = "block";
        main.style.width = "66.66666%";
        main.style.maxWidth = "100%";
        btn.innerHTML = '<i class="bi bi-caret-right-fill"></i>';
    }
    else {
        menu.style.display = "none";
        main.style.width = "100%";
        main.style.maxWidth = "";
        btn.innerHTML = '<i class="bi bi-caret-left-fill"></i>';
    }
    //small fix for super small viewport on mobile
    var vname = op + "_viewer";
    setTimeout(function(){tmapp[vname].viewport.applyConstraints(true);},100);
}


/** 
* @param {Object} options The id of the element
* @summary Create a complete new tab with all the UI, accordion and buttons. 
* Options are not implemented but are there if needed in the future 
*/
interfaceUtils.generateDataTabUI = function(options){
    var generated;
    if (options) if (options.uid) {
        interfaceUtils._mGenUIFuncs.ctx.aUUID = options.uid;
        generated=options.uid;
    }
    if (!generated) {
        interfaceUtils._mGenUIFuncs.generateUUID();
        generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;
    }
    if (! dataUtils.data[generated]) {
        divpane=interfaceUtils._mGenUIFuncs.generateTab();
        accordion=interfaceUtils._mGenUIFuncs.generateAccordion();
        
        //now that the 3 accordion items are created, fill tehm and 
        //add all to the corresponding main data tab

        item1rows=interfaceUtils._mGenUIFuncs.generateAccordionItem1();
        item1rows.forEach(row => accordion.contents[0].appendChild(row))

        item2rows=interfaceUtils._mGenUIFuncs.generateAccordionItem2();
        item2rows.forEach(row => accordion.contents[1].appendChild(row))

        item3rows=interfaceUtils._mGenUIFuncs.generateAccordionItem3();
        item3rows.forEach(row => accordion.contents[2].appendChild(row))

        buttonrow=interfaceUtils._mGenUIFuncs.generateRowOptionsButtons();

        menurow=interfaceUtils._mGenUIFuncs.rowForMarkerUI();

        togglerow=HTMLElementUtils.createElement({"kind":"div", "extraAttributes":{"class":"row"}});
        var divpane_settings_toggle = HTMLElementUtils.createElement({"kind":"div", "id":generated+"_marker-tab-settings-toggle", "extraAttributes":{"class":"d-none w-auto ms-auto btn btn-light btn-sm mx-3"}});
        divpane_settings_toggle.innerHTML = "<i class='bi bi-sliders'></i>";
        divpane_settings_toggle.addEventListener("click",function(event) {
            var divpane_settings = interfaceUtils.getElementById(generated+"_marker-tab-settings")
            divpane_settings.classList.remove("d-none");
            divpane_settings_toggle.classList.add("d-none");
        })
        var divpane_settings = HTMLElementUtils.createElement({"kind":"div", "id":generated+"_marker-tab-settings"});
        divpane_settings.appendChild(accordion.divaccordion);
        divpane_settings.appendChild(buttonrow);
        togglerow.append(divpane_settings_toggle)
            
        divpane.append(togglerow)

        //row progressbar
        row0=HTMLElementUtils.createRow({id:generated+"_csv_progress_parent"});
        row0.classList.add("d-none");
        row0.classList.add("px-3");
        row0.innerHTML="Loading markers..."

        col01=HTMLElementUtils.createColumn({"width":12});
            div011=HTMLElementUtils.createElement({"kind":"div", "extraAttributes":{"class":"progress"}});
                div0111=HTMLElementUtils.createElement({"kind":"div", "id":generated+"_csv_progress", "extraAttributes":{"class":"progress-bar progress-bar-striped progress-bar-animated","role":"progressbar" ,"aria-valuenow":"10", "aria-valuemin":"0" ,"aria-valuemax":"100"}});
        
        row0.appendChild(col01)
            col01.appendChild(div011)
                div011.appendChild(div0111);

        divpane.appendChild(row0);

        divpane.append(divpane_settings)
        divpane.appendChild(menurow);

        tabs1content=interfaceUtils.getElementById("level-1-tabsContent");
        if(tabs1content) tabs1content.appendChild(divpane);
        else { console.log("No level 1 tab content"); return;}
    }
    interfaceUtils._mGenUIFuncs.ActivateTab(generated);
    if (options) {
        console.log("options are here:");
        console.log(JSON.stringify(options));
        if (options.hideSettings) {
            divpane_settings = interfaceUtils.getElementById(generated+"_marker-tab-settings");
            divpane_settings.classList.add("d-none");
            divpane_settings_toggle = interfaceUtils.getElementById(generated+"_marker-tab-settings-toggle");
            divpane_settings_toggle.classList.remove("d-none");
        }
        interfaceUtils._mGenUIFuncs.ChangeTabName(generated, options.name);
        if (options.path !== undefined) {
            dataUtils.XHRCSV(generated,options);
        }
        if (options.expectedHeader === undefined) {
            $('#'+generated+'_flush-collapse0').collapse("show");
        }
        if (options.path !== undefined & options.fromButton === undefined) {
            projectUtils.makeButtonFromTabAux(generated, options.path, "", "", true);
        }
    }
    else {
        $('#'+generated+'_flush-collapse0').collapse("show");
    }
    return generated;
}

/**
 * @summary To not fill interfaceUtils with a lot of things, there is the _mGenUIFuncs
 * object encapsulating all the functions pertaining creation of tabs
 */
interfaceUtils._mGenUIFuncs={ctx:{aUUID:0}}

/** 
* @param {HTMLEvent} event event that triggered function
* @summary Delete all trace of a tab including datautils.data.key*/
interfaceUtils._mGenUIFuncs.deleteTab=function(uid){
    tabbutton=interfaceUtils.getElementById(uid+"_li-tab")
    if (!tabbutton) {return;}
    tabbutton.remove();

    tabpane=interfaceUtils.getElementById(uid+"_marker-pane")
    tabpane.remove();
    projectUtils.removeTabFromProject(uid);
    delete dataUtils.data[uid];

    glUtils.deleteMarkers(uid);
    glUtils.draw();
    tabButtons = interfaceUtils.getElementsByClassName("marker-tab-button")
    if (tabButtons.length > 0) {
        tabButtons[tabButtons.length - 1].click();
    }
}

/** 
* @param {Object} uid The id of the element
* @param {Object} drop The id of the dropdown element to convert
* @summary Converts a csv tab to h5 by replacing DropDown inputs with autocomplete 
*/
interfaceUtils._mGenUIFuncs.intputToH5 = function(uid, inputDropDown){
    if (!inputDropDown) return;
    //inputDropDown.parent.innerHTML = "";
    var inputText=HTMLElementUtils.createElement({"kind":"input", "id":inputDropDown.id, "extraAttributes":{ "name":inputDropDown.id, "class":"form-control","type":"text" }});
    if (inputDropDown.classList.contains("d-none"))
        inputText.classList.add("d-none");
    
    inputDropDown.parentNode.replaceChild(inputText, inputDropDown);
    let options = {
        tabDisabled: true,
        minChars: 0,
        appendTo: inputDropDown.parentNode,
        lookup: function (query, done) {
            // Do Ajax call or lookup locally, when done,
            // call the callback and pass your results:
            let url = dataUtils.data[uid]._csv_path
            dataUtils._hdf5Api.getKeys(url, query).then((data) => {
                let keys = data.children.map((value) => {
                    let completePath = value.replace("//","/");
                    return {"value": completePath, "data": completePath };
                },
                (error)=>{console.log("Error!",error)});
                var result = {
                    suggestions: keys
                };
                done(result);
            },function(error){console.log(error)})
        },
        onSelect: function (suggestion) {
            inputText.focus();
            const event = new Event('change');
            $("#" + inputDropDown.id)[0].dispatchEvent(event);
        }
    }
    $("#" + inputDropDown.id).autocomplete(options);
    return $("#" + inputDropDown.id)[0];
}

/** 
* @param {Object} uid The id of the element
* @summary Converts a csv tab to h5 by replacing DropDown inputs with autocomplete 
*/
interfaceUtils._mGenUIFuncs.dataTabUIToH5 = function(uid){
    var alldrops=interfaceUtils._mGenUIFuncs.getTabDropDowns(uid, true);
    var namesymbols=Object.getOwnPropertyNames(alldrops, true);
    namesymbols.forEach((drop)=>{
        interfaceUtils._mGenUIFuncs.intputToH5(uid, alldrops[drop]);
    })
}

/** 
* @param {HTMLEvent} event event that triggered function
* @param {Array.String} array domid suffixes within group
* @param {Array.Number} option this option will be shown while all others are hidden
* @summary This function takes options within one specific tab and hide all except the one marked by option */
interfaceUtils._mGenUIFuncs.hideShow=function(event,array,options){
    uid=event.target.id.split("_")[0]
    array.forEach((domid, index)=>{
        newdomid=uid+domid;
        domelement=interfaceUtils.getElementById(newdomid);
        if(domelement){
            if(options.includes(index)){
                domelement.classList.remove("d-none");
            }else{
                domelement.classList.add("d-none");
            }
        }
    });
}

/** 
* @param {HTMLEvent} event event that triggered function
* @param {Array.String} array domid suffixes within group
* @param {Number} option this option will be selected while all others are unselected
* @summary This function takes options within one specific tab and deselects all except the one marked by option */
interfaceUtils._mGenUIFuncs.selectDeselect=function(event,array,options){
    uid=event.target.id.split("_")[0]
    array.forEach((domid, index)=>{
        newdomid=uid+domid;
        domelement=interfaceUtils.getElementById(newdomid);
        if(domelement){
            if(options.includes(index)){
                domelement.checked=true;
            }else{
                domelement.checked=false;
            }
        }
    });
}

/** 
* @param {HTMLEvent} event event that triggered function
* @param {Array.String} array domid suffixes within group
* @param {Number} option this option will be enabled while all others are disabled
* @summary This function takes options within one specific tab and disables all except the one marked by option */
interfaceUtils._mGenUIFuncs.enableDisable=function(event,array,options){
    uid=event.target.id.split("_")[0];
    array.forEach((domid, index)=>{
        newdomid=uid+domid;
        domelement=interfaceUtils.getElementById(newdomid);
        //console.log(domelement,index,options,(index in options).toString())
        if(domelement){
            if(options.includes(index)){
                domelement.disabled=false;
            }else{
                domelement.disabled=true;
            }
        }
    });
}

/** 
* @param {HTMLEvent} event event that triggered function
* @summary Chages the name of the tab if this text in the form has changed */
interfaceUtils._mGenUIFuncs.ChangeTabName=function(uid, value){
    domelement=interfaceUtils.getElementById(uid+"_marker-tab-name");
    if(domelement){
        if(value)
            domelement.innerText=value
        else
            domelement.innerText=uid;
        domelement.setAttribute("title", domelement.innerText);
        interfaceUtils.getElementById(uid + "_tab-name").value = domelement.innerText;
    }
}

/** 
* @param {HTMLEvent} event event that triggered function
* @summary Chages the name of the tab if this text in the form has changed */
interfaceUtils._mGenUIFuncs.ActivateTab=function(uid){
    domelement=interfaceUtils.getElementById(uid+"_marker-tab-button");
    if(domelement){
        domelement.click();
    }
}

/**
 * @param {string} uid the data id
 * @summary Returns an object full with inputs for a tab named as: 
 * "X","Y","gb_sr","gb_col","gb_name","cb_cmap","cb_col"
 * @returns {Object} allinputs
 */
interfaceUtils._mGenUIFuncs.getTabDropDowns = function(uid, only_csvColumns){
    if (only_csvColumns === undefined) only_csvColumns = false;
    allinputs={}
    allinputs["X"]=interfaceUtils.getElementById(uid+"_x-value");
    allinputs["Y"]=interfaceUtils.getElementById(uid+"_y-value");

    allinputs["gb_col"]=interfaceUtils.getElementById(uid+"_gb-col-value");
    allinputs["gb_name"]=interfaceUtils.getElementById(uid+"_gb-col-name");

    allinputs["cb_col"]=interfaceUtils.getElementById(uid+"_cb-col-value");    

    allinputs["scale_col"]=interfaceUtils.getElementById(uid+"_scale-col");   
    allinputs["edges_col"]=interfaceUtils.getElementById(uid+"_edges-col");
    allinputs["sortby_col"]=interfaceUtils.getElementById(uid+"_sortby-col");
    allinputs["pie_col"]=interfaceUtils.getElementById(uid+"_piechart-col");
    allinputs["opacity_col"]=interfaceUtils.getElementById(uid+"_opacity-col");
    allinputs["shape_col"]=interfaceUtils.getElementById(uid+"_shape-col-value");
    allinputs["collectionItem_col"]=interfaceUtils.getElementById(uid+"_collectionItem-col-value");
    if (!only_csvColumns) {
        allinputs["z_order"]=interfaceUtils.getElementById(uid+"_z-order");
        allinputs["cb_gr_dict"]=interfaceUtils.getElementById(uid+"_cb-bygroup-dict-val");
        allinputs["cb_cmap"]=interfaceUtils.getElementById(uid+"_cb-cmap-value");
        allinputs["scale_factor"]=interfaceUtils.getElementById(uid+"_scale-factor");
        allinputs["coord_factor"]=interfaceUtils.getElementById(uid+"_coord-factor");
        allinputs["pie_dict"]=interfaceUtils.getElementById(uid+"_piechart-dict-val");
        allinputs["shape_fixed"]=interfaceUtils.getElementById(uid+"_shape-fixed-value");
        allinputs["shape_gr_dict"]=interfaceUtils.getElementById(uid+"_shape-bygroup-dict-val");
        allinputs["opacity"]=interfaceUtils.getElementById(uid+"_opacity");
        allinputs["tooltip_fmt"]=interfaceUtils.getElementById(uid+"_tooltip_fmt");
        allinputs["collectionItem_fixed"]=interfaceUtils.getElementById(uid+"_collectionItem-fixed-value");
    }
    return allinputs;
}

/**
 * @param {string} uid the data id
 * @summary Returns an object full with inputs for a tab named as: 
 * "gb_sr", "gb_col", "cb_cmap", "cb_col", "cb_gr", "cb_gr_rand", "cb_gr_gene", "cb_gr_name"
 * @returns {Object} allinputs
 */
interfaceUtils._mGenUIFuncs.getTabRadiosAndChecks= function(uid){
    allradios={}

    allradios["cb_col"]=interfaceUtils.getElementById(uid+"_cb-bypoint");
    allradios["cb_gr"]=interfaceUtils.getElementById(uid+"_cb-bygroup");

    allradios["cb_gr_rand"]=interfaceUtils.getElementById(uid+"_cb-bygroup-rand");
    allradios["cb_gr_dict"]=interfaceUtils.getElementById(uid+"_cb-bygroup-dict");
    allradios["cb_gr_key"]=interfaceUtils.getElementById(uid+"_cb-bygroup-key");

    allradios["pie_check"]=interfaceUtils.getElementById(uid+"_use-piecharts");
    allradios["edges_check"]=interfaceUtils.getElementById(uid+"_use-edges");
    allradios["sortby_check"]=interfaceUtils.getElementById(uid+"_use-sortby");
    allradios["sortby_desc_check"]=interfaceUtils.getElementById(uid+"_sortby-desc");
    allradios["scale_check"]=interfaceUtils.getElementById(uid+"_use-scales");
    allradios["shape_gr"]=interfaceUtils.getElementById(uid+"_shape-bygroup");
    allradios["shape_gr_rand"]=interfaceUtils.getElementById(uid+"_shape-bygroup-rand");
    allradios["shape_gr_dict"]=interfaceUtils.getElementById(uid+"_shape-bygroup-dict");
    allradios["shape_col"]=interfaceUtils.getElementById(uid+"_shape-bypoint");
    allradios["shape_fixed"]=interfaceUtils.getElementById(uid+"_shape-fixed");
    allradios["opacity_check"]=interfaceUtils.getElementById(uid+"_use-opacity");
    allradios["_no_outline"]=interfaceUtils.getElementById(uid+"__no-outline");
    allradios["collectionItem_col"]=interfaceUtils.getElementById(uid+"_collectionItem-bypoint");
    allradios["collectionItem_fixed"]=interfaceUtils.getElementById(uid+"_collectionItem-fixed");
    
    
    return allradios;
}

/**
 * @param {string} uid the data id
 * @summary Returns an object full with bools for checks and radios to see if they are checked
 * @returns {Object} allinputs
 */
 interfaceUtils._mGenUIFuncs.areRadiosAndChecksChecked = function(uid){

    var radios=interfaceUtils._mGenUIFuncs.getTabRadiosAndChecks(uid)
    var arechecked={};

    for(r in radios){
        arechecked[r]=radios[r].checked
    }
    
    return arechecked;
}


/**
 * Creates a unique id for each new tab 
 */
interfaceUtils._mGenUIFuncs.generateUUID=function(){
    //HAS TO START with letter
    //aUUID="U12345";
    aUUID='Uxxxxx'.replace(/[x]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
    });
    aUUID=aUUID.toUpperCase();  
    interfaceUtils._mGenUIFuncs.ctx.aUUID=aUUID;
}
   
/**
 * @summary Creates a bootstrap tab to put on the top of the menu and it's pane, 
 * the pane is returned so it can be added to the corresponding existing parent o all panes
 * @returns {HTMLElement} divpane
 */
interfaceUtils._mGenUIFuncs.generateTab=function(){
    //create the tab and the space for the content
    //fill context with generated value for ID of data type
    var generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

     /** 
     * TAB OBJECT
    */
    
    //first thing is to add the tab in the level 1. Which is a li with a button
    li1=HTMLElementUtils.createElement({"kind":"li", "id":generated+"_li-tab", "extraAttributes":{ "class":"nav-item", "role":"presentation"}});
    button1=HTMLElementUtils.createButton({"id":generated+"_marker-tab-button","extraAttributes":{ "class":"nav-link marker-tab-button", "data-bs-toggle":"tab","data-bs-target":"#"+generated+"_marker-pane","type":"button","role":"tab","aria-controls":generated+"_marker","aria-selected":"false"}})

    span1=HTMLElementUtils.createElement({"kind":"span", "id":generated+"_marker-tab-name","extraAttributes":{ "title": "New markers"}})
    span1.innerHTML="New markers";

    button1.appendChild(span1);
    closeButton=HTMLElementUtils.createElement({"kind":"a", "id":generated+"_marker-tab-close"})
    closeButton.innerHTML="&nbsp;&nbsp;<i class='bi bi-x'></i>";
    button1.appendChild(closeButton);
    closeButton.addEventListener("click",function(event) {
        interfaceUtils.confirm("Are you sure you want to delete this tab?")
        .then(function(_confirm){
            if (_confirm) interfaceUtils._mGenUIFuncs.deleteTab(generated)
        });
    })


    li1.appendChild(button1);
    ultabs1=interfaceUtils.getElementById("level-1-tabs");
    plusone=interfaceUtils.getElementById("plus-1");
    if(plusone && ultabs1) ultabs1.insertBefore(li1,plusone);
    else { console.log("No level 1 tabs"); return;}

    /** 
     * TAB PANE
    */
    //now the content of that tab pane which is a form like group to select the options for rendering
    //1.1
    divpane=HTMLElementUtils.createElement({"kind":"div", "id":generated+"_marker-pane", "extraAttributes":{  "class":"tab-pane",  "role":"tabpanel", "aria-labelledby":generated+"_marker-tab"}});

    //return this pane
    return divpane;
}

/**
 * @summary Generate the scaffold of the accordion. if you want more parts for the accordion, do it here.
 * Returns an object with two elements: Pointers to the accordion contents so that we can fill them with the correct forms
 * @returns {Object} divpane={divaccordion:_,contents:_}
 */
interfaceUtils._mGenUIFuncs.generateAccordion=function(){

    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;
    /** 
     * MAIN ACCORDION
    */
    //inside the pane put an accordion with 3 accordion-items to put the options
    divaccordion=HTMLElementUtils.createElement({"kind":"div","id":generated+"_accordion-flush","extraAttributes":{"class":"accordion accordion-flush"}})

    //now 3 accordion items
    accordionitems=[];
    accordioncontents=[];
    ["File and coordinates","Render options","Advanced options"].forEach(function(title,index){
        divaccordionitem=HTMLElementUtils.createElement({ "kind":"div","extraAttributes":{"class":"accordion-item"}});
        h2accordionitem=HTMLElementUtils.createElement({ "kind":"h2","id":"flush-heading"+index.toString(),"extraAttributes":{"class":"accordion-header"}});
        buttonaccordionitem=HTMLElementUtils.createElement({ "kind":"button", "extraAttributes":{ "class":"accordion-button collapsed", "type":"button", "data-bs-toggle":"collapse", "data-bs-target":"#"+generated+"_flush-collapse"+index.toString(), "aria-expanded":"false", "aria-controls":generated+"_flush-collapse"+index.toString()}})
        divaccordioncontent=HTMLElementUtils.createElement({ "kind":"div", "id":generated+"_flush-collapse"+index.toString(), "extraAttributes":{ "class":"accordion-collapse collapse tm-accordion-collapse py-2", "data-bs-parent":"#"+generated+"_accordion-flush", "aria-labelledby":"flush-heading"+index.toString()}})
        buttonaccordionitem.innerText=title;

        h2accordionitem.appendChild(buttonaccordionitem);
        divaccordionitem.appendChild(h2accordionitem);
        divaccordionitem.appendChild(divaccordioncontent);

        accordionitems.push(divaccordionitem);
        accordioncontents.push(divaccordioncontent);
    })

    accordionitems.forEach(ait =>{divaccordion.appendChild(ait)});

    //return pointers to the accordion contents so that we can 
    //fill them with the correct forms
    return {divaccordion:divaccordion,contents:accordioncontents};

}

 /**
 * @summary Creates progrwss bar, input file picker, tab name, X and Y and returns rows to append to the accordion
 * @returns {array} array of rows
 */
interfaceUtils._mGenUIFuncs.generateAccordionItem1=function(){

    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;
    
    
    //row 1
    row1=HTMLElementUtils.createRow({id:generated+"_row-1"});
        col11=HTMLElementUtils.createColumn({"width":6, "id":generated+"_input_csv_col"});
            div111=HTMLElementUtils.createElement({"kind":"div", "id":generated+"_input_csv"});
                label1111=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"for":generated+"_csv"}});
                label1111.innerText="File and coordinates";
                input1112=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_csv","extraAttributes":{ "name":generated+"_csv", 
                "class":"form-control-file form-control form-control-sm", "type":"file", "accept":".csv,.tsv,.txt,.h5,.h5ad"}});
                input1112.addEventListener("change",(event)=>{dataUtils.startCSVcascade(event)});
    
    //---------------------------------

    col12=HTMLElementUtils.createColumn({"width":6});
        div121=HTMLElementUtils.createElement({"kind":"div","id":"input-group"});
            label1221=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"for":generated+"_tab-name"}});
            label1221.innerText="Tab name";
            input1222=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_tab-name", "extraAttributes":{ "name":generated+"_tab-name", "class":"form-control","type":"text", "placeholder":"New markers", "value":"New markers","aria-label":"Tab name" }});
            input1222.innerText=generated; 
            input1222.addEventListener("change",(event)=>{interfaceUtils._mGenUIFuncs.ChangeTabName(event.target.name.split("_")[0], event.target.value);})

    ///ROW 2

    row2=HTMLElementUtils.createRow({"id":generated+"_row-2"});
        col21=HTMLElementUtils.createColumn({"width":6});
            label211=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_x-label", "extraAttributes":{ "for":generated+"_x-value" }});
            label211.innerText="X coordinate"
            select212=HTMLElementUtils.createElement({"kind":"select", "id":generated+"_x-value", "extraAttributes":{ "class":"form-select form-select-sm", "aria-label":".form-select-sm"}});

        col22=HTMLElementUtils.createColumn({"width":6});
            label221=HTMLElementUtils.createElement({"kind":"label","id":generated+"_y-label","extraAttributes":{"for":generated+"_y-value" }});
            label221.innerText="Y coordinate";
            select222=HTMLElementUtils.createElement({"kind":"select", "id":generated+"_y-value", "extraAttributes":{ "class":"form-select form-select-sm", "aria-label":".form-select-sm"} });

    row3=HTMLElementUtils.createRow({"id":generated+"_row-3"});
        col30=HTMLElementUtils.createColumn({"width":4});
            label301=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_coord-factor"}});
            label301.innerHTML="Coordinate scale factor";
            inputscalefactor=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_coord-factor","extraAttributes":{ "class":"form-text-input", "type":"number", "value":1, "min":0, "step":0.05}});            

    /*row0.appendChild(col01)
        col01.appendChild(div011)
            div011.appendChild(div0111);*/

    row1.appendChild(col11);
        col11.appendChild(div111);  
            div111.appendChild(label1111);
            div111.appendChild(input1112);
    row1.appendChild(col12);
        col12.appendChild(div121);
            div121.appendChild(label1221);
            div121.appendChild(input1222);

    row2.appendChild(col21);
        col21.appendChild(label211);
        col21.appendChild(select212);
    row2.appendChild(col22);    
        col22.appendChild(label221);
        col22.appendChild(select222);

    row3.appendChild(col30);
        col30.appendChild(label301);
        col30.appendChild(inputscalefactor);


    return [row1,row2,row3];

}

 /**
 * @summary Creates the forms to color by
 * @returns {array} array of rows
 */
interfaceUtils._mGenUIFuncs.generateColorByAccordion2= function(){
    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    ///col 1

    //------------------------------------
    rowcb=HTMLElementUtils.createRow({"id":generated+"_colorby"});

    colcb1=HTMLElementUtils.createColumn({"width":12});

    labelcb11=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_cb-label"});
    labelcb11.innerHTML="<strong>Color options</strong>";


    //col 2
    //-----------------------------------

    colcb2=HTMLElementUtils.createColumn({"width":4});
        divformcheck1cb=HTMLElementUtils.createElement({"kind":"div","extraAttributes":{"class":"form-check"}});
            inputradio1cb=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_cb-bygroup","extraAttributes":{ "name":generated+"_flexRadioColorBy", "class":"form-check-input", "type":"radio", "checked":true}});
            labelcbgroup=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_cb-bygroup"}});
            labelcbgroup.innerText="Color by group";
        
        divformcheck2cb=HTMLElementUtils.createElement({"kind":"div", "extraAttributes":{"class":"form-check"}});
            inputradio2cb=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_cb-bypoint","extraAttributes":{"name":generated+"_flexRadioColorBy","class":"form-check-input","type":"radio"}});
            labelcbpoint=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_cb-bypoint"}});
            labelcbpoint.innerText="Color by marker";
   
    //------------------------

    colcb3=HTMLElementUtils.createColumn({"width":8});
        //create a whole group for color by group, random, key and group name
        divoptionscolgroup=HTMLElementUtils.createElement({"kind":"div","id":generated+"_cb-col-group-options","extraAttributes":{"class": "renderOptionContainer"}});

            rowkey=HTMLElementUtils.createElement({"kind":"div","id":generated+"_row-cb-gr-key","extraAttributes":{"class": "form-check"}});
                inputradiocbgrkey=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_cb-bygroup-key","extraAttributes":{ "name":generated+"_flexRadioColorByGroup", "class":"form-check-input", "type":"radio", "checked":true}});
                labelcbgroupkey=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_cb-bygroup-key"}});
                labelcbgroupkey.innerHTML="Generate color from key value<br>";

            rowrand=HTMLElementUtils.createElement({"kind":"div","id":generated+"_row-cb-gr-rand","extraAttributes":{"class": "form-check"}});
                inputradiocbgrrand=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_cb-bygroup-rand","extraAttributes":{ "name":generated+"_flexRadioColorByGroup", "class":"form-check-input", "type":"radio"}});
                labelcbgrouprand=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_cb-bygroup-rand"}});
                labelcbgrouprand.innerHTML="Generate color randomly<br>";

            rowdict=HTMLElementUtils.createElement({"kind":"div","id":generated+"_row-cb-gr-dict","extraAttributes":{"class": "form-check"}});
                inputradiocbgrdict=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_cb-bygroup-dict","extraAttributes":{ "name":generated+"_flexRadioColorByGroup", "class":"form-check-input", "type":"radio"}});
                labelcbgroupdict=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_cb-bygroup-dict"}});
                labelcbgroupdict.innerHTML="Use color from dictionary<br>";
                inputtextcbgrdict=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_cb-bygroup-dict-val","extraAttributes":{ "class":"form-text-input", "type":"text", "placeholder":"{'key1':''#FFFFFF',...}"}});
                inputtextcbgrdict.disabled=true

        divoptionscol=HTMLElementUtils.createElement({"kind":"div","id":generated+"_cb-col-options","extraAttributes":{"class": "renderOptionContainer d-none"}});
            selectcbcol=HTMLElementUtils.createElement({"kind":"select","id":generated+"_cb-col-value","extraAttributes":{"class":"form-select form-select-sm","aria-label":".form-select-sm"}});
            labelcbcol=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_cb_col-colname-label","extraAttributes":{"for":generated+"_cb-col-value"} });
            labelcbcol.innerText="Select color column";
        divoptionscmap=HTMLElementUtils.createElement({"kind":"div", "id":generated+"_cb-cmap-options","extraAttributes":{"class": "renderOptionContainer d-none"}});
            labelcbcmapvalue=HTMLElementUtils.createElement({"kind":"label","id":generated+"_cb-cmap-label","extraAttributes":{"for":generated+"_cb-cmap-value"}});
            labelcbcmapvalue.innerText="Color map (only if color column is numeral)";
            cmapoptions=[{"text":"None","value":""}];
            dataUtils._d3LUTs.forEach((lut)=>{ cmapoptions.push({"text":lut.replace("interpolate",""),"value":lut}) })
            selectcbcmap=HTMLElementUtils.selectTypeDropDown({ "id":generated+"_cb-cmap-value","class":"form-select form-select-sm","options":cmapoptions,"extraAttributes":{"aria-label":".form-select-sm"}})

    //listeners

    inputradio1cb.addEventListener("change",(event)=>{
        interfaceUtils._mGenUIFuncs.hideShow(event,["_cb-cmap-options","_cb-col-options","_cb-col-group-options"],[2])
    });
    inputradio2cb.addEventListener("change",(event)=>{
        interfaceUtils._mGenUIFuncs.hideShow(event,["_cb-cmap-options","_cb-col-options","_cb-col-group-options"],[0,1])
    });
    inputradiocbgrdict.addEventListener("change",(event)=>{
        interfaceUtils._mGenUIFuncs.enableDisable(event,["_cb-bygroup-dict-val"],[0])
    });
    inputradiocbgrrand.addEventListener("change",(event)=>{
        interfaceUtils._mGenUIFuncs.enableDisable(event,["_cb-bygroup-dict-val"],[])
    });
    inputradiocbgrkey.addEventListener("change",(event)=>{
        interfaceUtils._mGenUIFuncs.enableDisable(event,["_cb-bygroup-dict-val"],[])
    });

    rowcb.appendChild(colcb1);
        colcb1.appendChild(labelcb11);
    rowcb.appendChild(colcb2);
        colcb2.appendChild(divformcheck1cb);
            divformcheck1cb.appendChild(inputradio1cb);
            divformcheck1cb.appendChild(labelcbgroup);
        colcb2.appendChild(divformcheck2cb);
            divformcheck2cb.appendChild(inputradio2cb);
            divformcheck2cb.appendChild(labelcbpoint);
    rowcb.appendChild(colcb3);
        colcb3.appendChild(divoptionscolgroup);    
            divoptionscolgroup.appendChild(rowkey);
                rowkey.appendChild(inputradiocbgrkey);
                rowkey.appendChild(labelcbgroupkey);   
            divoptionscolgroup.appendChild(rowrand);
                rowrand.appendChild(inputradiocbgrrand);
                rowrand.appendChild(labelcbgrouprand);        
            divoptionscolgroup.appendChild(rowdict);
                rowdict.appendChild(inputradiocbgrdict);
                rowdict.appendChild(labelcbgroupdict);
                rowdict.appendChild(inputtextcbgrdict);
        colcb3.appendChild(divoptionscol);
            divoptionscol.appendChild(labelcbcol);
            divoptionscol.appendChild(selectcbcol);
        colcb3.appendChild(divoptionscmap);
            divoptionscmap.appendChild(labelcbcmapvalue);
            divoptionscmap.appendChild(selectcbcmap);

    return rowcb;
    
}

 /**
 * @summary Creates the forms to group by
 * @returns {array} a single rows
 */
  interfaceUtils._mGenUIFuncs.generateKeyColAccordion2= function(){
    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    //row 0
    row0=HTMLElementUtils.createRow({id:generated+"_key_0"});
        collab=HTMLElementUtils.createColumn({"width":12});
            labellab=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_cb-label"});
            labellab.innerHTML="<strong>Group by</strong>";

        col00=HTMLElementUtils.createColumn({"width":6});
            label010=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_key-col-label", "extraAttributes":{ "for":generated+"_key-col" }});
            label010.innerText="Key to group by (optional)"
            select011=HTMLElementUtils.createElement({"kind":"select", "id":generated+"_gb-col-value", "extraAttributes":{ "class":"form-select form-select-sm", "aria-label":".form-select-sm"}});
            select011.disabled=false
            
            label012=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_name-col-label", "extraAttributes":{ "for":generated+"_name-col" }});
            label012.innerText="Extra column to display (optional)"
            select013=HTMLElementUtils.createElement({"kind":"select", "id":generated+"_gb-col-name", "extraAttributes":{ "class":"form-select form-select-sm", "aria-label":".form-select-sm"}});
            select013.disabled=false

    row0.appendChild(collab)
        collab.appendChild(labellab)

    row0.appendChild(col00);
        col00.appendChild(label010);
        col00.appendChild(select011);
        col00.appendChild(label012);
        col00.appendChild(select013);


    return row0;
}

 /**
 * @summary Creates the whole options section
 * @returns {array} array of rows
 */
interfaceUtils._mGenUIFuncs.generateAccordionItem2=function(){

    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;
    
    row1=interfaceUtils._mGenUIFuncs.generateKeyColAccordion2();
    row2=interfaceUtils._mGenUIFuncs.generateColorByAccordion2();

    return [row1, row2];
}

/**
 * @summary Creates advanced options
 * @returns {array} array of rows
 */
 interfaceUtils._mGenUIFuncs.generateAccordionItem3=function(){

    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    row1=interfaceUtils._mGenUIFuncs.generateAdvancedScaleAccordion3();
    row2=interfaceUtils._mGenUIFuncs.generateAdvancedPiechartAccordion3();
    row3=interfaceUtils._mGenUIFuncs.generateAdvancedEdgesAccordion3();
    row4=interfaceUtils._mGenUIFuncs.generateAdvancedShapeAccordion3();
    row5=interfaceUtils._mGenUIFuncs.generateAdvancedOpacityAccordion3();
    row6=interfaceUtils._mGenUIFuncs.generateAdvancedSortbyAccordion3();
    row7=interfaceUtils._mGenUIFuncs.generateAdvancedTooltipAccordion3();
    row8=interfaceUtils._mGenUIFuncs.generateAdvancedCollectionAccordion3();
    row9=interfaceUtils._mGenUIFuncs.generateAdvancedMakeButtonAccordion3();
    
    return [row1,row2,row3,row4,row5,row6,row7,row8,row9];
 }

 /**
 * @summary Creates the forms to scale by
 * @returns {array} a single rows
 */
    interfaceUtils._mGenUIFuncs.generateAdvancedScaleAccordion3= function(){
    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    //row 0
    row0=HTMLElementUtils.createRow({id:generated+"_scale_0"});
        collab=HTMLElementUtils.createColumn({"width":12});
            labellab=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_cb-label"});
            labellab.innerHTML="<strong>Marker size</strong>";

        col00=HTMLElementUtils.createColumn({"width":6});
            label0002=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_scale-factor"}});
            label0002.innerHTML="Size factor:&nbsp;";
            inputsizefactor=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_scale-factor","extraAttributes":{ "class":"form-text-input", "type":"number", "value":1, "min":0, "step":0.05}});
            divformcheck000=HTMLElementUtils.createElement({ "kind":"div", "extraAttributes":{"class":"form-check"}});
                inputcheck0000=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_use-scales","extraAttributes":{"class":"form-check-input","type":"checkbox" }});
                label0001=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_use-scales-label", "extraAttributes":{ "for":generated+"_use-scales" }});
                label0001.innerText="Use different size per marker"
            
                
        col01=HTMLElementUtils.createColumn({"width":6});
            label010=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_scale-col-label", "extraAttributes":{ "for":generated+"_scale-col", "class":"d-none" }});
            label010.innerText="Size column"
            select011=HTMLElementUtils.createElement({"kind":"select", "id":generated+"_scale-col", "extraAttributes":{ "class":"form-select form-select-sm d-none", "aria-label":".form-select-sm"}});
            select011.disabled=true

    inputcheck0000.addEventListener("change", (event)=>{
        var value=event.target.checked;
        //var doms=["_gb-single","_gb-col","_gb-feature-value","_cb-colormap","_cb-bypoint","_cb-bygroup","_gb-feature-value",
        //          "_gb-col-value","_gb-col-name","_cb-cmap-value","_cb-col-value","_cb-bygroup-rand","_cb-bygroup-gene","_cb-bygroup-name" ]
        if(value) {
            interfaceUtils._mGenUIFuncs.enableDisable(event, ["_scale-col"],[0])
            interfaceUtils._mGenUIFuncs.hideShow(event, ["_scale-col","_scale-col-label"],[0,1])
        }
        else { 
            interfaceUtils._mGenUIFuncs.enableDisable(event, ["_scale-col"],[])
            interfaceUtils._mGenUIFuncs.hideShow(event, ["_scale-col","_scale-col-label"],[])
        }
    })

    row0.appendChild(collab)
        collab.appendChild(labellab)

    row0.appendChild(col00)
        col00.appendChild(label0002);
        col00.appendChild(inputsizefactor);
        col00.appendChild(divformcheck000)
            divformcheck000.appendChild(inputcheck0000);
            divformcheck000.appendChild(label0001);

    row0.appendChild(col01);
        col01.appendChild(label010);
        col01.appendChild(select011);


    return row0;
}

 /**
 * @summary Creates the forms to shape by
 * @returns {array} a single rows
 */
  interfaceUtils._mGenUIFuncs.generateAdvancedShapeAccordion3= function(){
    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    //row 0
    row0=HTMLElementUtils.createRow({id:generated+"_shape_0"});
        collab=HTMLElementUtils.createColumn({"width":12});
            labellab=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_shape-label"});
            labellab.innerHTML="<strong>Marker shape</strong>";

        colshape2=HTMLElementUtils.createColumn({"width":6});
            divformcheck1shape=HTMLElementUtils.createElement({"kind":"div","extraAttributes":{"class":"form-check"}});
                inputradio1shape=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_shape-bygroup","extraAttributes":{ "name":generated+"_flexRadioShapeBy", "class":"form-check-input", "type":"radio", "checked":true}});
                labelshapegroup=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_shape-bygroup"}});
                labelshapegroup.innerText="Shape by group";
            
            divformcheck2shape=HTMLElementUtils.createElement({"kind":"div", "extraAttributes":{"class":"form-check"}});
                inputradio2shape=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_shape-bypoint","extraAttributes":{"name":generated+"_flexRadioShapeBy","class":"form-check-input","type":"radio"}});
                labelshapepoint=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_shape-bypoint"}});
                labelshapepoint.innerText="Shape by marker";
            
            divformcheck3shape=HTMLElementUtils.createElement({"kind":"div", "extraAttributes":{"class":"form-check"}});
                inputradio3shape=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_shape-fixed","extraAttributes":{"name":generated+"_flexRadioShapeBy","class":"form-check-input","type":"radio"}});
                labelshapefixed=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_shape-fixed"}});
                labelshapefixed.innerText="Use a fixed shape";
        
            divformcheck4shape=HTMLElementUtils.createElement({ "kind":"div", "extraAttributes":{"class":"form-check"}});
                inputcheck4shape=HTMLElementUtils.createElement({"kind":"input", "id":generated+"__no-outline","extraAttributes":{"class":"form-check-input","type":"checkbox" }});
                label4shape=HTMLElementUtils.createElement({"kind":"label", "id":generated+"__no-outline-label", "extraAttributes":{ "for":generated+"__no-outline" }});
                label4shape.innerText="Remove Outline"
        //------------------------
    
        colshape3=HTMLElementUtils.createColumn({"width":6});
            //create a whole group for shape by group, random, key and group name
            divoptionscolgroup=HTMLElementUtils.createElement({"kind":"div","id":generated+"_shape-col-group-options","extraAttributes":{"class": "renderOptionContainer"}});
    
                rowrand=HTMLElementUtils.createElement({"kind":"div","id":generated+"_row-shape-gr-rand","extraAttributes":{"class": "form-check"}});
                    inputradioshapegrrand=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_shape-bygroup-rand","extraAttributes":{ "name":generated+"_flexRadioShapeByGroup", "class":"form-check-input", "type":"radio", "checked":true}});
                    labelshapegrouprand=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_shape-bygroup-rand"}});
                    labelshapegrouprand.innerHTML="Select shape iteratively<br>";
    
                rowdict=HTMLElementUtils.createElement({"kind":"div","id":generated+"_row-shape-gr-dict","extraAttributes":{"class": "form-check"}});
                    inputradioshapegrdict=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_shape-bygroup-dict","extraAttributes":{ "name":generated+"_flexRadioShapeByGroup", "class":"form-check-input", "type":"radio"}});
                    labelshapegroupdict=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_shape-bygroup-dict"}});
                    labelshapegroupdict.innerHTML="Use shape from dictionary<br>";
                    inputtextshapegrdict=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_shape-bygroup-dict-val","extraAttributes":{ "class":"form-text-input", "type":"text", "placeholder":"{'key1':''#FFFFFF',...}"}});
                    inputtextshapegrdict.disabled=true
    
            divoptionscol=HTMLElementUtils.createElement({"kind":"div","id":generated+"_shape-col-options","extraAttributes":{"class": "renderOptionContainer d-none"}});
                selectshapecol=HTMLElementUtils.createElement({"kind":"select","id":generated+"_shape-col-value","extraAttributes":{"class":"form-select form-select-sm","aria-label":".form-select-sm"}});
                labelshapecol=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_shape_col-colname-label","extraAttributes":{"for":generated+"_shape-col-value"} });
                labelshapecol.innerText="Select shape column";
            
            divoptionsfixed=HTMLElementUtils.createElement({"kind":"div","id":generated+"_shape-fixed-options","extraAttributes":{"class": "renderOptionContainer d-none"}});
                labelfixedshapevalue=HTMLElementUtils.createElement({"kind":"label","id":generated+"_shape-fixed-label","extraAttributes":{"for":generated+"_shape-fixed-value"}});
                labelfixedshapevalue.innerText="Select shape";
                shapeoptions=[];
                markerUtils._symbolStrings.forEach((sho, shoIndex)=>{ shapeoptions.push({"text":markerUtils._symbolUnicodes[shoIndex],"value":sho}) })
                shapeinput2=HTMLElementUtils.selectTypeDropDown({ "id":generated+"_shape-fixed-value","class":"form-select form-select-sm","options":shapeoptions,"extraAttributes":{"aria-label":".form-select-sm"}})
                shapeinput2.value=markerUtils._symbolStrings[0]


        //listeners

    inputradio1shape.addEventListener("change",(event)=>{
        interfaceUtils._mGenUIFuncs.hideShow(event,["_shape-col-options","_shape-col-group-options","_shape-fixed-options"],[1])
    });
    inputradio2shape.addEventListener("change",(event)=>{
        interfaceUtils._mGenUIFuncs.hideShow(event,["_shape-col-options","_shape-col-group-options","_shape-fixed-options"],[0])
    });
    inputradio3shape.addEventListener("change",(event)=>{
        interfaceUtils._mGenUIFuncs.hideShow(event,["_shape-col-options","_shape-col-group-options","_shape-fixed-options"],[2])
    });
    inputradioshapegrdict.addEventListener("change",(event)=>{
        interfaceUtils._mGenUIFuncs.enableDisable(event,["_shape-bygroup-dict-val"],[0])
    });
    inputradioshapegrrand.addEventListener("change",(event)=>{
        interfaceUtils._mGenUIFuncs.enableDisable(event,["_shape-bygroup-dict-val"],[])
    });
    
    row0.appendChild(collab)
        collab.appendChild(labellab)

    row0.appendChild(colshape2);
        colshape2.appendChild(divformcheck1shape);
            divformcheck1shape.appendChild(inputradio1shape);
            divformcheck1shape.appendChild(labelshapegroup);
        colshape2.appendChild(divformcheck2shape);
            divformcheck2shape.appendChild(inputradio2shape);
            divformcheck2shape.appendChild(labelshapepoint);
        colshape2.appendChild(divformcheck3shape);
            divformcheck3shape.appendChild(inputradio3shape);
            divformcheck3shape.appendChild(labelshapefixed);
        colshape2.appendChild(divformcheck4shape);
            divformcheck4shape.appendChild(inputcheck4shape);
            divformcheck4shape.appendChild(label4shape);
    row0.appendChild(colshape3);
        colshape3.appendChild(divoptionscolgroup);    
            divoptionscolgroup.appendChild(rowrand);
                rowrand.appendChild(inputradioshapegrrand);
                rowrand.appendChild(labelshapegrouprand);        
            divoptionscolgroup.appendChild(rowdict);
                rowdict.appendChild(inputradioshapegrdict);
                rowdict.appendChild(labelshapegroupdict);
                rowdict.appendChild(inputtextshapegrdict);
        colshape3.appendChild(divoptionscol);
            divoptionscol.appendChild(labelshapecol);
            divoptionscol.appendChild(selectshapecol);
        colshape3.appendChild(divoptionsfixed);
            divoptionsfixed.appendChild(labelfixedshapevalue);
            divoptionsfixed.appendChild(shapeinput2);

    return row0;
}

 /**
 * @summary Creates the forms for collection id
 * @returns {array} a single rows
 */
  interfaceUtils._mGenUIFuncs.generateAdvancedCollectionAccordion3= function(){
    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    //row 0
    row0=HTMLElementUtils.createRow({id:generated+"_collectionItem_0"});
        collab=HTMLElementUtils.createColumn({"width":12});
            labellab=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_collectionItem-label"});
            labellab.innerHTML="<strong>Collection mode</strong>";

        colcollectionItem2=HTMLElementUtils.createColumn({"width":6});
            
            divformcheck2collectionItem=HTMLElementUtils.createElement({"kind":"div", "extraAttributes":{"class":"form-check"}});
                inputradio2collectionItem=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_collectionItem-fixed","extraAttributes":{"name":generated+"_flexRadioCollectionBy","class":"form-check-input","type":"radio", "checked":true}});
                labelcollectionItemfixed=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_collectionItem-fixed"}});
                labelcollectionItemfixed.innerText="Use a fixed collection item";

            divformcheck3collectionItem=HTMLElementUtils.createElement({"kind":"div", "extraAttributes":{"class":"form-check"}});
                inputradio3collectionItem=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_collectionItem-bypoint","extraAttributes":{"name":generated+"_flexRadioCollectionBy","class":"form-check-input","type":"radio"}});
                labelcollectionItempoint=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_collectionItem-bypoint"}});
                labelcollectionItempoint.innerText="Collection item by marker";
        //------------------------
    
        colcollectionItem3=HTMLElementUtils.createColumn({"width":6});
            //create a whole group for collectionItem by group, random, key and group name
            
            divoptionsfixed=HTMLElementUtils.createElement({"kind":"div","id":generated+"_collectionItem-fixed-options","extraAttributes":{"class": "renderOptionContainer"}});
                labelfixedcollectionItemvalue=HTMLElementUtils.createElement({"kind":"label","id":generated+"_collectionItem-fixed-label","extraAttributes":{"for":generated+"_collectionItem-fixed-value"}});
                labelfixedcollectionItemvalue.innerText="Specify collection item index";
                collectionIteminput2=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_collectionItem-fixed-value","extraAttributes":{ "class":"form-text-input", "type":"number", "value":0, "step":1, "min":0, "max":1000}});

            divoptionscol=HTMLElementUtils.createElement({"kind":"div","id":generated+"_collectionItem-col-options","extraAttributes":{"class": "renderOptionContainer d-none"}});
                selectcollectionItemcol=HTMLElementUtils.createElement({"kind":"select","id":generated+"_collectionItem-col-value","extraAttributes":{"class":"form-select form-select-sm","aria-label":".form-select-sm"}});
                labelcollectionItemcol=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_collectionItem_col-colname-label","extraAttributes":{"for":generated+"_collectionItem-col-value"} });
                labelcollectionItemcol.innerText="Select collection item index column";
            

        //listeners

    inputradio2collectionItem.addEventListener("change",(event)=>{
        interfaceUtils._mGenUIFuncs.hideShow(event,["_collectionItem-col-options","_collectionItem-col-group-options","_collectionItem-fixed-options"],[2])
    });
    inputradio3collectionItem.addEventListener("change",(event)=>{
        interfaceUtils._mGenUIFuncs.hideShow(event,["_collectionItem-col-options","_collectionItem-col-group-options","_collectionItem-fixed-options"],[0])
    });
    
    row0.appendChild(collab)
        collab.appendChild(labellab)

    row0.appendChild(colcollectionItem2);
        colcollectionItem2.appendChild(divformcheck2collectionItem);
            divformcheck2collectionItem.appendChild(inputradio2collectionItem);
            divformcheck2collectionItem.appendChild(labelcollectionItemfixed);
        colcollectionItem2.appendChild(divformcheck3collectionItem);
            divformcheck3collectionItem.appendChild(inputradio3collectionItem);
            divformcheck3collectionItem.appendChild(labelcollectionItempoint);
    row0.appendChild(colcollectionItem3);
        colcollectionItem3.appendChild(divoptionsfixed);
            divoptionsfixed.appendChild(labelfixedcollectionItemvalue);
            divoptionsfixed.appendChild(collectionIteminput2);
        colcollectionItem3.appendChild(divoptionscol);
            divoptionscol.appendChild(labelcollectionItemcol);
            divoptionscol.appendChild(selectcollectionItemcol);
        
    return row0;
}

 /**
 * @summary Creates the forms for piecharts
 * @returns {array} a single rows
 */
  interfaceUtils._mGenUIFuncs.generateAdvancedPiechartAccordion3= function(){
    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    //row 0
    row0=HTMLElementUtils.createRow({id:generated+"_piechart_0"});
        collab=HTMLElementUtils.createColumn({"width":12});
            labellab=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_cb-label"});
            labellab.innerHTML="<strong>Pie-charts</strong>";

        col00=HTMLElementUtils.createColumn({"width":6});
            divformcheck000=HTMLElementUtils.createElement({ "kind":"div", "extraAttributes":{"class":"form-check"}});
                inputcheck0000=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_use-piecharts","extraAttributes":{"class":"form-check-input","type":"checkbox" }});
                label0001=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_use-piecharts-label", "extraAttributes":{ "for":generated+"_use-piecharts" }});
                label0001.innerText="Use pie-charts"
                
        col01=HTMLElementUtils.createColumn({"width":6});
            label010=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_piechart-col-label", "extraAttributes":{ "class":"d-none", "for":generated+"_piechart-col" }});
            label010.innerText="Pie-chart column"
            select011=HTMLElementUtils.createElement({"kind":"select", "id":generated+"_piechart-col", "extraAttributes":{ "class":"d-none form-select form-select-sm", "aria-label":".form-select-sm"}});
            select011.disabled=true
            label012=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_piechart-dict-label", "extraAttributes":{ "class":"d-none", "for":generated+"_piechart-dict" }});
            label012.innerText="Pie-chart colors"
            input013=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_piechart-dict-val","extraAttributes":{ "class":"d-none form-text-input", "type":"text", "placeholder":"{'key1':''#FFFFFF',...}"}});            

    inputcheck0000.addEventListener("change", (event)=>{
        var value=event.target.checked;
        //var doms=["_gb-single","_gb-col","_gb-feature-value","_cb-colormap","_cb-bypoint","_cb-bygroup","_gb-feature-value",
        //          "_gb-col-value","_gb-col-name","_cb-cmap-value","_cb-col-value","_cb-bygroup-rand","_cb-bygroup-gene","_cb-bygroup-name" ]
        if(value){
            interfaceUtils._mGenUIFuncs.enableDisable(event, ["_piechart-col","_piechart-dict-val","_cb-bygroup","_cb-bypoint","_shape-bygroup","_shape-bypoint","_shape-fixed","_shape-bygroup-rand","_shape-bygroup-dict","_shape-col-value","_shape-fixed-value","_shape-col","_cb-bygroup-key","_cb-bygroup-rand","_cb-bygroup-dict"],[0,1]);
            interfaceUtils._mGenUIFuncs.hideShow(event, ["_piechart-col-label","_piechart-col","_piechart-dict-label","_piechart-dict-val"],[0,1,2,3]);
        }
        else {
            interfaceUtils._mGenUIFuncs.enableDisable(event, ["_piechart-col","_piechart-dict-val","_cb-bygroup","_cb-bypoint","_shape-bygroup","_shape-bypoint","_shape-fixed","_shape-bygroup-rand","_shape-bygroup-dict","_shape-col-value","_shape-fixed-value","_cb-bygroup-key","_cb-bygroup-rand","_cb-bygroup-dict"],[2,3,4,5,6,7,8,9,10,11,12,13]);
            interfaceUtils._mGenUIFuncs.hideShow(event, ["_piechart-col-label","_piechart-col","_piechart-dict-label","_piechart-dict-val"],[]);
        }
    })

    row0.appendChild(collab)
        collab.appendChild(labellab)

    row0.appendChild(col00)
        col00.appendChild(divformcheck000)
            divformcheck000.appendChild(inputcheck0000);
            divformcheck000.appendChild(label0001);

    row0.appendChild(col01);
        col01.appendChild(label010);
        col01.appendChild(select011);
        col01.appendChild(label012);
        col01.appendChild(input013);

    return row0;
}

 /**
 * @summary Creates the forms for edges
 * @returns {array} a single rows
 */
  interfaceUtils._mGenUIFuncs.generateAdvancedEdgesAccordion3= function(){
    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    //row 0
    row0=HTMLElementUtils.createRow({id:generated+"_edges_0"});
        collab=HTMLElementUtils.createColumn({"width":12});
            labellab=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_cb-label"});
            labellab.innerHTML="<strong>Network diagram</strong>";

        col00=HTMLElementUtils.createColumn({"width":6});
            divformcheck000=HTMLElementUtils.createElement({ "kind":"div", "extraAttributes":{"class":"form-check"}});
                inputcheck0000=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_use-edges","extraAttributes":{"class":"form-check-input","type":"checkbox" }});
                label0001=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_use-edges-label", "extraAttributes":{ "for":generated+"_use-edges" }});
                label0001.innerText="Add Edges"
                
        col01=HTMLElementUtils.createColumn({"width":6});
            label010=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_edges-col-label", "extraAttributes":{ "class":"d-none", "for":generated+"_edges-col" }});
            label010.innerText="Edges column"
            select011=HTMLElementUtils.createElement({"kind":"select", "id":generated+"_edges-col", "extraAttributes":{ "class":"d-none form-select form-select-sm", "aria-label":".form-select-sm"}});

    inputcheck0000.addEventListener("change", (event)=>{
        var value=event.target.checked;
        //var doms=["_gb-single","_gb-col","_gb-feature-value","_cb-colormap","_cb-bypoint","_cb-bygroup","_gb-feature-value",
        //          "_gb-col-value","_gb-col-name","_cb-cmap-value","_cb-col-value","_cb-bygroup-rand","_cb-bygroup-gene","_cb-bygroup-name" ]
        if(value){
            interfaceUtils._mGenUIFuncs.hideShow(event, ["_edges-col-label","_edges-col","_edges-dict-label","_edges-dict-val"],[0,1,2,3]);
        }
        else {
            interfaceUtils._mGenUIFuncs.hideShow(event, ["_edges-col-label","_edges-col","_edges-dict-label","_edges-dict-val"],[]);
        }
    })

    row0.appendChild(collab)
        collab.appendChild(labellab)

    row0.appendChild(col00)
        col00.appendChild(divformcheck000)
            divformcheck000.appendChild(inputcheck0000);
            divformcheck000.appendChild(label0001);

    row0.appendChild(col01);
        col01.appendChild(label010);
        col01.appendChild(select011);

    return row0;
}

 /**
 * @summary Creates the forms for opacity
 * @returns {array} a single rows
 */
  interfaceUtils._mGenUIFuncs.generateAdvancedOpacityAccordion3= function(){
    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    //row 0
    row0=HTMLElementUtils.createRow({id:generated+"_opacity_0"});
        collab=HTMLElementUtils.createColumn({"width":12});
            labellab=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_cb-label"});
            labellab.innerHTML="<strong>Marker opacity</strong>";

        col00=HTMLElementUtils.createColumn({"width":6});
            label0002=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_opacity"}});
            label0002.innerHTML="Opacity value:&nbsp;";
            inputsizefactor=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_opacity","extraAttributes":{ "class":"form-text-input", "type":"number", "value":1, "step":0.05, "min":0, "max":1}});
            divformcheck000=HTMLElementUtils.createElement({ "kind":"div", "extraAttributes":{"class":"form-check"}});
                inputcheck0000=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_use-opacity","extraAttributes":{"class":"form-check-input","type":"checkbox" }});
                label0001=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_use-opacity-label", "extraAttributes":{ "for":generated+"_use-opacity" }});
                label0001.innerText="Use different opacity per marker"

        col01=HTMLElementUtils.createColumn({"width":6});
            label010=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_opacity-col-label", "extraAttributes":{ "for":generated+"_opacity-col", "class":"d-none" }});
            label010.innerText="Opacity column"
            select011=HTMLElementUtils.createElement({"kind":"select", "id":generated+"_opacity-col", "extraAttributes":{ "class":"form-select form-select-sm d-none", "aria-label":".form-select-sm"}});
            select011.disabled=true

    inputcheck0000.addEventListener("change", (event)=>{
        var value=event.target.checked;
        if(value) {
            interfaceUtils._mGenUIFuncs.enableDisable(event, ["_opacity-col"],[0])
            interfaceUtils._mGenUIFuncs.hideShow(event, ["_opacity-col","_opacity-col-label"],[0,1])
        }
        else {
            interfaceUtils._mGenUIFuncs.enableDisable(event, ["_opacity-col"],[])
            interfaceUtils._mGenUIFuncs.hideShow(event, ["_opacity-col","_opacity-col-label"],[])
        }
    })
            
    row0.appendChild(collab)
        collab.appendChild(labellab)

    row0.appendChild(col00)
        col00.appendChild(label0002);
        col00.appendChild(inputsizefactor);
        col00.appendChild(divformcheck000)
            divformcheck000.appendChild(inputcheck0000);
            divformcheck000.appendChild(label0001);

    row0.appendChild(col01);
        col01.appendChild(label010);
        col01.appendChild(select011);

    return row0;
}

 /**
 * @summary Creates the forms for sorting
 * @returns {array} a single rows
 */
 interfaceUtils._mGenUIFuncs.generateAdvancedSortbyAccordion3= function(){
    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    //row 0
    row0=HTMLElementUtils.createRow({id:generated+"_sortby_0"});
        collab=HTMLElementUtils.createColumn({"width":12});
            labellab=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_cb-label"});
            labellab.innerHTML="<strong>Marker ordering</strong>";

        col00=HTMLElementUtils.createColumn({"width":6});
            divformcheck000=HTMLElementUtils.createElement({ "kind":"div", "extraAttributes":{"class":"form-check"}});
                inputcheck0000=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_use-sortby","extraAttributes":{"class":"form-check-input","type":"checkbox" }});
                label0001=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_use-sortby-label", "extraAttributes":{ "for":generated+"_use-sortby" }});
                label0001.innerText="Sort markers"
                
        col01=HTMLElementUtils.createColumn({"width":6});
            label010=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_sortby-col-label", "extraAttributes":{ "class":"d-none", "for":generated+"_sortby-col" }});
            label010.innerText="Sort by column"
            select011=HTMLElementUtils.createElement({"kind":"select", "id":generated+"_sortby-col", "extraAttributes":{ "class":"d-none form-select form-select-sm", "aria-label":".form-select-sm"}});
            divformcheck012=HTMLElementUtils.createElement({ "kind":"div", "id":generated+"_sortby-desc-div", "extraAttributes":{"class":"d-none form-check"}});
                inputcheck0120=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_sortby-desc","extraAttributes":{"class":"form-check-input","type":"checkbox" }});
                label0121=HTMLElementUtils.createElement({"kind":"label", "id":generated+"sortby-desc-label", "extraAttributes":{ "for":generated+"_sortby-desc" }});
                label0121.innerText="Use descending order"

        col10=HTMLElementUtils.createColumn({"width":6});
            label101=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_z-order"}});
            label101.innerHTML="Z-order value:&nbsp;";
            input102=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_z-order","extraAttributes":{ "class":"form-text-input", "type":"number", "value":1, "step":0.05, "min":0, "max":1}});
        
    inputcheck0000.addEventListener("change", (event)=>{
        var value=event.target.checked;
        //var doms=["_gb-single","_gb-col","_gb-feature-value","_cb-colormap","_cb-bypoint","_cb-bygroup","_gb-feature-value",
        //          "_gb-col-value","_gb-col-name","_cb-cmap-value","_cb-col-value","_cb-bygroup-rand","_cb-bygroup-gene","_cb-bygroup-name" ]
        if(value){
            interfaceUtils._mGenUIFuncs.hideShow(event, ["_sortby-desc-div", "_sortby-col-label","_sortby-col","_sortby-dict-label","_sortby-dict-val"],[0,1,2,3]);
        }
        else {
            interfaceUtils._mGenUIFuncs.hideShow(event, ["_sortby-desc-div", "_sortby-col-label","_sortby-col","_sortby-dict-label","_sortby-dict-val"],[]);
        }
    })

    row0.appendChild(collab)
        collab.appendChild(labellab)

    row0.appendChild(col00)
        col00.appendChild(divformcheck000)
            divformcheck000.appendChild(inputcheck0000);
            divformcheck000.appendChild(label0001);

    row0.appendChild(col01);
        col01.appendChild(label010);
        col01.appendChild(select011);
        col01.appendChild(divformcheck012);
            divformcheck012.appendChild(inputcheck0120);
            divformcheck012.appendChild(label0121);

    row0.appendChild(col10)
        col10.appendChild(label101);
        col10.appendChild(input102);
    
    return row0;
}

 /**
 * @summary Creates the forms to scale by
 * @returns {array} a single rows
 */
  interfaceUtils._mGenUIFuncs.generateAdvancedTooltipAccordion3= function(){
    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    //row 0
    row0=HTMLElementUtils.createRow({id:generated+"_tooltip_0"});
        collab=HTMLElementUtils.createColumn({"width":12});
            labellab=HTMLElementUtils.createElement({"kind":"label", "id":generated+"_cb-label"});
            labellab.innerHTML="<strong>Marker tooltip</strong>";

        col00=HTMLElementUtils.createColumn({"width":6});
            label0002=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":generated+"_tooltip_fmt"}});
            label0002.innerHTML="Format:&nbsp;";
            inputsizefactor=HTMLElementUtils.createElement({"kind":"input", "id":generated+"_tooltip_fmt","extraAttributes":{ "class":"form-text-input", "type":"text", "value":""}});
            
    row0.appendChild(collab)
        collab.appendChild(labellab)

    row0.appendChild(col00)
        col00.appendChild(label0002);
        col00.appendChild(inputsizefactor);

    return row0;
}

 /**
 * @summary Creates the forms to scale by
 * @returns {array} a single rows
 */
  interfaceUtils._mGenUIFuncs.generateAdvancedMakeButtonAccordion3= function(){
    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    var generateButton = function (event) {
        return projectUtils.makeButtonFromTab(event.target.id.split("_")[0]);
    } 
    //row 0
    row0=HTMLElementUtils.createRow({id:generated+"_opacity_0"});
        col00=HTMLElementUtils.createColumn({"width":6});
            button000=HTMLElementUtils.createButton({
                "id":generated+"_Generate-button-from-tab",
                "innerText":"Generate button from tab",
                "class":"btn btn-light my-1",
                "eventListeners":{"click":generateButton }
            });
            
    row0.appendChild(col00)
        col00.appendChild(button000);

    return row0;
}

 /**
 * @summary Creates the forms to color by
 * @returns {HTMLElement} row
 */
interfaceUtils._mGenUIFuncs.generateRowOptionsButtons=function(){
    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;
    row0=HTMLElementUtils.createRow({"id":generated+"_row-option-buttons"});
    row0.classList.add("updateViewRow")
        col00=HTMLElementUtils.createColumn({"width":8});
        //col01=HTMLElementUtils.createColumn({"width":3});
        //    button010=HTMLElementUtils.createButton({"id":generated+"_delete-button","innerText":"Close tab","class":"btn btn-secondary","eventListeners":{"click":(event)=>interfaceUtils._mGenUIFuncs.deleteTab(event)}});
        col02=HTMLElementUtils.createColumn({"width":4});
            button020=HTMLElementUtils.createButton({"id":generated+"_update-view-button","innerText":"Update view","class":" btn btn-primary my-1","eventListeners":{"click":(event)=> dataUtils.updateViewOptions(event.target.id.split("_")[0]) }});
    
    row0.appendChild(col00);
    //row0.appendChild(col01);
    //    col01.appendChild(button010);
    row0.appendChild(col02);
        col02.appendChild(button020);

    return row0;

}

interfaceUtils._mGenUIFuncs.rowForMarkerUI=function(){

    generated=interfaceUtils._mGenUIFuncs.ctx.aUUID;

    row0=HTMLElementUtils.createRow({id:generated+"_menu-UI"});
    row0.classList.add("d-none");

    return row0;
}

/**
 * @param {string} uid id in datautils.data
 * @param {object} expectedHeader object of the type {input:expectedHeaderFromCSV}
 * @summary If somehow there is an expect CSV this will help you del with that
*/
interfaceUtils._mGenUIFuncs.fillDropDownsIfExpectedCSV=function(uid,expectedHeader){
    //expected headr is an object that has these keys, other will be ignored;
    //"X","Y","gb_sr","gb_col","gb_name","cb_cmap","cb_col"
    if(expectedHeader){
        dropdowns=interfaceUtils._mGenUIFuncs.getTabDropDowns(uid);

        for(d in expectedHeader){
            if(dropdowns[d]){
                needle=expectedHeader[d];
                if (typeof needle === 'object') needle = JSON.stringify(needle);
                opts=dropdowns[d].options;
                if (!opts) {
                    dropdowns[d].value=needle
                }
                else {
                    for(var i=0;i<opts.length;i++){
                        var o=opts[i];
                        proceed=o.value.includes(needle) 
                        if(proceed){
                            dropdowns[d].value=needle
                        }
                    }
                }          
            }
        }
    }
}

/**
 * @param {string} uid id in datautils.data
 * @param {object} expectedRadios object of the type {input:expectedHeaderFromCSV}
 * @summary If somehow there is an expect CSV this will help you del with that
*/
interfaceUtils._mGenUIFuncs.fillRadiosAndChecksIfExpectedCSV=function(uid,expectedRadios){
    //expected headr is an object that has these keys, other will be ignored;
    //"X","Y","gb_sr","gb_col","gb_name","cb_cmap","cb_col"
    if(expectedRadios){
        var radios=interfaceUtils._mGenUIFuncs.getTabRadiosAndChecks(uid);
        for(var d in expectedRadios){
            if(radios[d]){
                needle=expectedRadios[d];
                radios[d].checked=needle;
                if (needle) {
                    var event = new Event('change');
                    radios[d].dispatchEvent(event);
                }
            }
        }
    }
}

/**
 * @param {string} uid id in datautils.data
 * @summary Create the menu with the options to select marker, select shape and color to draw
*/
interfaceUtils._mGenUIFuncs.groupUI=async function(uid, force){
    //if we arrive here it's because  agroupgarden exists, all the information is there, 
    //also we need some info on color and options, but we can get that.
    var data_obj = dataUtils.data[uid];
    
    if (force === undefined && Object.keys(data_obj["_groupgarden"]).length > 3000) {
        let _confirm = await interfaceUtils.confirm("You are trying to load " + Object.keys(data_obj["_groupgarden"]).length + " different groups, which can be slow and make TissUUmaps unresponsive. Are you sure you want to continue?","Warning")
        if (_confirm) {
            return interfaceUtils._mGenUIFuncs.groupUI(uid, true);
        }
        else {
            interfaceUtils._mGenUIFuncs.deleteTab(uid, true);
            return null;
        }
    }

    var _selectedOptions=interfaceUtils._mGenUIFuncs.areRadiosAndChecksChecked(uid);
    var _selectedDropDown=interfaceUtils._mGenUIFuncs.getTabDropDowns(uid);

    //I do this to know if I have name selected, and also to know where to draw the 
    //color from
    var groupUI=HTMLElementUtils.createElement({"kind":"div"});
    var filter=HTMLElementUtils.createElement({"kind":"input", "extraAttributes":{ "class":"form-text-input form-control", "type":"text", "placeholder":"Filter markers"}});

    groupUI.appendChild(filter)

    var table=HTMLElementUtils.createElement({"kind":"table","extraAttributes":{"class":"table table-striped marker_table"}});
    var thead=HTMLElementUtils.createElement({"kind":"thead"});
    var thead2=HTMLElementUtils.createElement({"kind":"thead"});
    var theadrow=HTMLElementUtils.createElement({"kind":"tr"});
    var tbody=HTMLElementUtils.createElement({"kind":"tbody"});

    var headopts=[""];
    var sortable = {}
    if(data_obj["_gb_col"]){
        headopts.push(data_obj["_gb_col"]);
        sortable[data_obj["_gb_col"]] = "sorttable_sort";
    }
    else { 
        headopts.push("Group");
        sortable["Group"] = "sorttable_nosort";
    }
    
    var usename=false;
    if(data_obj["_gb_name"]){
        headopts.push(data_obj["_gb_name"]);
        sortable[data_obj["_gb_name"]] = "sorttable_sort";
        usename=true;
    }
    headopts.push("Counts");
    sortable["Counts"] = "sorttable_sort";
    if(!data_obj["_shape_col"] && !data_obj["_pie_col"]){
        headopts.push("Shape");
        sortable["Shape"] = "sorttable_nosort";
    }
    if(!data_obj["_cb_col"] && !data_obj["_pie_col"]){
        headopts.push("Color");
        sortable["Color"] = "sorttable_nosort";
    }
    headopts.push("");
    sortable[""] = "sorttable_nosort";
    headopts.forEach((opt)=>{
        var th=HTMLElementUtils.createElement({"kind":"th","extraAttributes":{"scope":"col","class":sortable[opt]}});
        th.innerText=opt
        theadrow.appendChild(th);
    });

    thead.appendChild(theadrow);

    if(data_obj["_gb_col"]){
        // add All row
        
        //row
        var tr=HTMLElementUtils.createElement({"kind":"tr","extraAttributes":{"class":"sorttable_nosort"}});
        //first spot for a check
        var td0=HTMLElementUtils.createElement({"kind":"td"});
        var td1=HTMLElementUtils.createElement({"kind":"td"});
        var td15=null;
        var td17=HTMLElementUtils.createElement({"kind":"td"});
        var td2=null;
        var td3=null;
        var td4=HTMLElementUtils.createElement({"kind":"td"});

        tr.appendChild(td0);
        tr.appendChild(td1);

        // Get previous "All" checkbox element so that we can re-use its old state
        const lastCheckAll = interfaceUtils.getElementById(uid+"_all_check", false);

        var check0=HTMLElementUtils.createElement({"kind":"input", "id":uid+"_all_check","extraAttributes":{"class":"form-check-input","type":"checkbox" }});
        check0.checked = lastCheckAll != null ? lastCheckAll.checked : true;
        td0.appendChild(check0);
        check0.addEventListener("input",(event)=>{
            clist = interfaceUtils.getElementsByClassName(uid+"-marker-input");
            for (var i = 0; i < clist.length; ++i) { clist[i].checked = event.target.checked; }
        });
        var label1=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"for":uid+"_all_check","class":"cursor-pointer"}});
        label1.innerText="All";
        td1.appendChild(label1);

        if(usename){
            var td15=HTMLElementUtils.createElement({"kind":"td"});
            var label15=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"for":uid+"_all_check","class":"cursor-pointer"}});
            label15.innerText="All";
            td15.appendChild(label15);
            tr.appendChild(td15);
        }

        var label17=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"for":uid+"_all_check","class":"cursor-pointer"}});
        //label17.innerText=data_obj["_processeddata"].length;    
        label17.innerText=data_obj["_processeddata"][data_obj["_X"]].length;  // FIXME
        td17.appendChild(label17);        
        tr.appendChild(td17);

        if(!data_obj["_shape_col"] && !data_obj["_pie_col"]){
            var td2=HTMLElementUtils.createElement({"kind":"td"});
            tr.appendChild(td2);
        }
        if(!data_obj["_cb_col"] && !data_obj["_pie_col"]){
            var td3=HTMLElementUtils.createElement({"kind":"td"});
            tr.appendChild(td3);
        }
        tr.appendChild(td4);
        thead2.appendChild(tr);
    }

    var countShape=0;
    var countColor=0;
    var favouriteShapes = [6,0,2,1,3,4,10,5]
    for(i of Object.keys(data_obj["_groupgarden"]).sort()){

        var tree = data_obj["_groupgarden"][i]
        
        //remove space just in case
        var escapedID=tree["treeID"].replace(/ /g,"_");
        var escapedName="";
        if(usename)
            escapedName=tree["treeName"];

        //row
        var tr=HTMLElementUtils.createElement({"kind":"tr",extraAttributes:{"data-uid":uid,"data-escapedID":escapedID,"data-key":tree["treeID"]}});
        //first spot for a check
        var td0=HTMLElementUtils.createElement({"kind":"td"});
        var td1=HTMLElementUtils.createElement({"kind":"td"});
        var td15=null;
        var td17=HTMLElementUtils.createElement({"kind":"td",extraAttributes:{"sorttable_customkey":-dataUtils._quadtreeSize(tree)}});
        var td2=null;
        var td3=null;
        var td4=HTMLElementUtils.createElement({"kind":"td"});

        tr.appendChild(td0);
        tr.appendChild(td1);

        // Get previous group checkbox elements so that we can re-use their old state
        const lastCheck0 = interfaceUtils.getElementById(uid+"_"+escapedID+"_check", false);
        const lastCheck1 = interfaceUtils.getElementById(uid+"_"+escapedID+"_hidden", false);

        var check0=HTMLElementUtils.createElement({"kind":"input", "id":uid+"_"+escapedID+"_check","extraAttributes":{"class":"form-check-input "+uid+"-marker-input","type":"checkbox" }});
        check0.checked = lastCheck0 != null ? lastCheck0.checked : true;
        td0.appendChild(check0);
        
        var check1=HTMLElementUtils.createElement({"kind":"input", "id":uid+"_"+escapedID+"_hidden","extraAttributes":{"class":"form-check-input marker-hidden d-none "+uid+"-marker-hidden","type":"checkbox" }});
        check1.checked = lastCheck1 != null ? lastCheck1.checked : false;
        td0.appendChild(check1);
        
        var label1=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"for":uid+"_"+escapedID+"_check","class":"cursor-pointer"}});
        label1.innerText=tree["treeID"];
        td1.appendChild(label1);

        if(usename){
            var td15=HTMLElementUtils.createElement({"kind":"td"});
            var label15=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"for":uid+"_"+escapedID+"_check","class":"cursor-pointer"}});
            label15.innerText=tree["treeName"];    
            td15.appendChild(label15);        
            tr.appendChild(td15);
        }

        var label17=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"for":uid+"_"+escapedID+"_check","class":"cursor-pointer"}});
        //label17.innerText=tree.size();    
        label17.innerText=dataUtils._quadtreeSize(tree);
        td17.appendChild(label17);        
        tr.appendChild(td17);

        if(!data_obj["_shape_col"] && !data_obj["_pie_col"]){
            td2 = HTMLElementUtils.createElement({"kind":"td"});
            var shapeoptions=[];
            markerUtils._symbolStrings.forEach((sho,index)=>{ shapeoptions.push({"text":markerUtils._symbolUnicodes[index],"value":sho}) })
            shapeinput2=HTMLElementUtils.selectTypeDropDown({ "id":uid+"_"+escapedID+"_shape","class":"form-select form-select-sm "+uid+"-marker-shape","options":shapeoptions,"extraAttributes":{"aria-label":".form-select-sm"}})
            if(_selectedOptions["shape_fixed"]){
                shapeinput2.value=_selectedDropDown["shape_fixed"].value;
            }else if(_selectedOptions["shape_gr_rand"]){
                shapeinput2.value=markerUtils._symbolStrings[favouriteShapes[countShape]];
                countShape+=1;
                countShape=countShape % favouriteShapes.length;
            }else if(_selectedOptions["shape_gr_dict"]){
                try {
                    val = JSON.parse(_selectedDropDown["shape_gr_dict"].value)[tree["treeID"]];
                    if (val !== undefined) {
                        shapeinput2.value=val;
                    }
                    else {
                        shapeinput2.value=markerUtils._symbolStrings[favouriteShapes[countShape]];
                        countShape+=1;
                        countShape=countShape % favouriteShapes.length;
                    }
                }
                catch (err){
                    shapeinput2.value=markerUtils._symbolStrings[favouriteShapes[countShape]];
                    countShape+=1;
                    countShape=countShape % favouriteShapes.length;
                }
            }
            shapeinput2.addEventListener("change",(event)=>{
                interfaceUtils.updateShapeDict(uid);
            });
            tr.appendChild(td2);
            td2.appendChild(shapeinput2);
        }
        if(!data_obj["_cb_col"] && !data_obj["_pie_col"]){
            td3 = HTMLElementUtils.createElement({"kind":"td"});
            //the color depends on 3 possibilities , "cb_gr_rand","cb_gr_gene","cb_gr_name"
            if(_selectedOptions["cb_gr_rand"]){
                thecolor=overlayUtils.randomColor("hex");
            }else if(_selectedOptions["cb_gr_key"]){
                thecolor=HTMLElementUtils.determinsticHTMLColor(escapedID);
            }else if(_selectedOptions["cb_gr_dict"]){
                try {
                    colorObject = JSON.parse(_selectedDropDown["cb_gr_dict"].value)
                    if (Array.isArray(colorObject)) {
                        thecolor=colorObject[countColor % colorObject.length];
                        countColor += 1;
                    }
                    else if (typeof colorObject === "object") {
                        thecolor=colorObject[tree["treeID"]];
                    }
                    if (thecolor === undefined) {
                        thecolor=HTMLElementUtils.determinsticHTMLColor(escapedID);
                    }
                }
                catch (err){
                    thecolor=overlayUtils.randomColor("hex");
                }
            }
            thecolor = thecolor.toLowerCase();  // Should be lowercase for color inputs
            var colorinput3 = HTMLElementUtils.inputTypeColor({"id": uid+"_"+escapedID+"_color", "class":uid+"-marker-color", "extraAttributes": {"value": thecolor}});
            tr.appendChild(td3);
            td3.appendChild(colorinput3);
            // fix for Safari
            colorinput3.value = "#ffffff";
            colorinput3.value = thecolor;

            colorinput3.addEventListener("change",(event)=>{
                interfaceUtils.updateColorDict(uid);
            });
        }

        
        button1 = HTMLElementUtils.createElement({"kind":"div", extraAttributes:{"data-uid":uid,"data-escapedID":escapedID, "class":"btn btn-light btn-sm mx-1"}});
        button1.innerHTML = "<i class='bi bi-eye'></i>";
        button1.checkVisible = check0;
        button1.checkHidden = check1;
        // Store this state so that we can also "preview" unchecked marker groups
        button1.lastCheckedState = check0.checked;
        td4.appendChild(button1);
        tr.appendChild(td4);
               
        eventnames = ["mouseenter"];
        eventnames.forEach(function(eventname) {
            button1.addEventListener("mouseenter",function(event) {
                tr = this.parentElement.parentElement;
                tr.classList.add("table-primary");
                hidden_inputs = interfaceUtils.getElementsByClassName("marker-hidden");
                for(var i = 0; i < hidden_inputs.length; i++){
                    hidden_inputs[i].checked = true;
                }
                this.lastCheckedState = this.checkVisible.checked;
                this.checkVisible.checked = true;
                this.checkHidden.checked = false;
                glUtils.updateColorLUTTextures();
                glUtils.draw();
            })
        })
        eventnames = ["mouseleave"];
        eventnames.forEach(function(eventname){
            button1.addEventListener(eventname,function(event) {
                tr = this.parentElement.parentElement;
                tr.classList.remove("table-primary");
                hidden_inputs = interfaceUtils.getElementsByClassName("marker-hidden");
                for(var i = 0; i < hidden_inputs.length; i++){
                    hidden_inputs[i].checked = false;
                }
                this.checkVisible.checked = this.lastCheckedState;  // Restore visible checkbox
                glUtils.updateColorLUTTextures();
                glUtils.draw();
            })
        })

        tbody.appendChild(tr);
       
    }

    table.appendChild(thead);
    table.appendChild(thead2);
    table.appendChild(tbody);
    groupUI.appendChild(table);
    filter.addEventListener("input",function(event) {
        // Temporary hides the table for chrome issue with slowliness
        table.classList.add("d-none");
        const trs = table.querySelectorAll('tbody tr')
        const filter = this.value
        const regex = new RegExp(filter, 'i')
        const isFoundInTds = td => regex.test(td.innerText)
        const isFound = childrenArr => childrenArr.some(isFoundInTds)
        const setTrStyleDisplay = (element) => {
            if (isFound([
            ...element.children // <-- All columns
            ])) {
                element.classList.remove("d-none");
            }
            else {
                element.classList.add("d-none");
            }
        }
        trs.forEach(setTrStyleDisplay);
        // Shows back the table
        table.classList.remove("d-none");
    })
    
    sorttable.makeSortable(table);
    if(data_obj["_gb_col"]){
        var myTH = table.getElementsByTagName("th")[1];
        sorttable.innerSortFunction.apply(myTH, []);
    }
    return groupUI;
}

interfaceUtils.updateColorDict = function(uid) {
    var data_obj = dataUtils.data[uid];
    jsonDict = {};
    for(i in data_obj["_groupgarden"]){
        var tree = data_obj["_groupgarden"][i]
        var escapedID=tree["treeID"].replace(/ /g,"_");
        var colorInput = interfaceUtils.getElementById(uid+"_"+escapedID+"_color");
        jsonDict[tree["treeID"]] = colorInput.value;
    }
    var colorDictInput = interfaceUtils.getElementById(uid+"_cb-bygroup-dict-val");
    colorDictInput.value = JSON.stringify(jsonDict);
    var colorDictRadio = interfaceUtils.getElementById(uid+"_cb-bygroup-dict");
    colorDictRadio.checked = true;
    colorDictInput.disabled = false;
};

interfaceUtils.updateShapeDict = function(uid) {
    var data_obj = dataUtils.data[uid];
    jsonDict = {};
    for(i in data_obj["_groupgarden"]){
        var tree = data_obj["_groupgarden"][i]
        var escapedID=tree["treeID"].replace(/ /g,"_");
        var shapeInput = interfaceUtils.getElementById(uid+"_"+escapedID+"_shape");
        jsonDict[tree["treeID"]] = shapeInput.value;
    }
    var shapeDictInput = interfaceUtils.getElementById(uid+"_shape-bygroup-dict-val");
    shapeDictInput.value = JSON.stringify(jsonDict);
    var colorDictRadio1 = interfaceUtils.getElementById(uid+"_shape-bygroup");
    colorDictRadio1.click();
    var colorDictRadio2 = interfaceUtils.getElementById(uid+"_shape-bygroup-dict");
    colorDictRadio2.checked = true;
    shapeDictInput.disabled = false;
};

interfaceUtils._mGenUIFuncs.getGroupInputs = function(uid, key) {
    const data_obj = dataUtils.data[uid];

    let inputs = {};
    if (data_obj["_groupgarden"].hasOwnProperty(key)) {
        const tree = data_obj["_groupgarden"][key];
        const escapedID = tree["treeID"].replace(/ /g,"_");
        // Assume that element for visibility checkbox always exists in the UI
        const hasGroupUI = interfaceUtils.getElementById(uid + "_" + escapedID + "_check");

        if (hasGroupUI) {
            inputs["visible"] = interfaceUtils.getElementById(uid + "_" + escapedID + "_check").checked;
            inputs["hidden"] = interfaceUtils.getElementById(uid + "_" + escapedID + "_hidden").checked;
            if (interfaceUtils.getElementById(uid + "_" + escapedID + "_shape", false))
                inputs["shape"] = interfaceUtils.getElementById(uid + "_" + escapedID + "_shape").value;
            if (interfaceUtils.getElementById(uid + "_" + escapedID + "_color", false))
                inputs["color"] = interfaceUtils.getElementById(uid + "_" + escapedID + "_color").value;
        }
    }
    return inputs;
}

interfaceUtils.loadingModal = function(text, title) {
    if (!title) title = "Loading";
    if (!text) text = "Please wait...";
    var modalUID = "loading"
    
    buttons=divpane=HTMLElementUtils.createElement({"kind":"div"});
    content=HTMLElementUtils.createElement({"kind":"p", "extraAttributes":{"class":""}});
    content.innerHTML = text;
    return interfaceUtils.generateModal(title, content, buttons, modalUID, true);
}

interfaceUtils.alert = function(text, title) {
    if (!title) title = "Alert";
    var modalUID = "messagebox"
    button1=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-primary mx-2", "data-bs-dismiss":"modal"}})
    button1.innerText = "Ok";
    buttons=divpane=HTMLElementUtils.createElement({"kind":"div"});
    buttons.appendChild(button1);
    button1.addEventListener("click",function(event) {
        $(`#${modalUID}_modal`).modal('hide');
    })
    content=HTMLElementUtils.createElement({"kind":"p", "extraAttributes":{"class":""}});
    content.innerHTML = text;
    interfaceUtils.generateModal(title, content, buttons, modalUID);
}

interfaceUtils.confirm = function (text, title) {
    return new Promise((resolve, reject) => {
        if (!title) title = "Confirm";
        var modalUID = "messagebox"
        button1=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-primary mx-2"}})
        button1.innerText = "Yes";
        button2=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-secondary mx-2", "data-bs-dismiss":"modal"}})
        button2.innerText = "No";
        buttons=divpane=HTMLElementUtils.createElement({"kind":"div"});
        buttons.appendChild(button1);
        buttons.appendChild(button2);
        button1.addEventListener("click",function(event) {
            $(`#${modalUID}_modal`).modal('hide');;
            resolve(true);
        })
        button2.addEventListener("click",function(event) {
            $(`#${modalUID}_modal`).modal('hide');;
            resolve(false);
        })
        content=HTMLElementUtils.createElement({"kind":"p", "extraAttributes":{"class":""}});
        content.innerHTML = text;
        interfaceUtils.generateModal(title, content, buttons, modalUID);
    })
}

interfaceUtils.prompt = function (text, value, title, type) {
    return new Promise((resolve, reject) => {
        if (!title) title = "Prompt";
        var modalUID = "messagebox"
        button1=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-primary mx-2"}})
        button1.innerText = "Ok";
        button2=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-secondary mx-2", "data-bs-dismiss":"modal"}})
        button2.innerText = "Cancel";
        buttons=divpane=HTMLElementUtils.createElement({"kind":"div"});
        buttons.appendChild(button1);
        buttons.appendChild(button2);
        button1.addEventListener("click",function(event) {
            $(`#${modalUID}_modal`).modal('hide');;
            resolve(document.getElementById("confirmModalValue").value);
        })
        button2.addEventListener("click",function(event) {
            $(`#${modalUID}_modal`).modal('hide');;
            reject();
        })
        content=HTMLElementUtils.createElement({"kind":"div"});
            
        row0=HTMLElementUtils.createElement({"kind":"p", "extraAttributes":{"class":""}});
        row0.innerHTML = text
        row1=HTMLElementUtils.createRow({});
            col11=HTMLElementUtils.createColumn({"width":12});
                if (type === undefined) {
                    input112=HTMLElementUtils.createElement({"kind":"input", "id":"confirmModalValue", "extraAttributes":{ "class":"form-text-input form-control", "type":"text", "value":value}});
                }
                else {
                    input112=HTMLElementUtils.createElement({"kind":"input", "id":"confirmModalValue", "extraAttributes":{ "class":"form-text-input form-control", "type":type, "value":value}});
                }

        content.appendChild(row0);
        content.appendChild(row1);
            row1.appendChild(col11);
                col11.appendChild(input112);
        interfaceUtils.generateModal(title, content, buttons, modalUID);
        input112.focus();
        input112.select();
    })
}

interfaceUtils.generateModal = function(title, content, buttons, uid, noClose) {
    if (!noClose) noClose = false;
    if (!uid) uid = "default";
    modalWindow = document.getElementById(uid + "_modal");
    if (! modalWindow) {
        var div = HTMLElementUtils.createElement({"kind":"div", "id":uid+"_modal", "extraAttributes":{ "class":"modal in fade", "tabindex":"-1", "role":"dialog", "aria-hidden":"true", "tabindex":"-1"}});
        div.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header cursor-pointer">
                        <h5 class="modal-title" id="${uid}_modalTitle"></h5>
                        <button type="button" id="${uid}_closeButton" class="btn-close" data-bs-dismiss="modal" aria-label="Close" onclick="$('#${uid}_modal').modal('hide');;"></button>
                    </div>
                    <div class="modal-body" id="${uid}_modalContent">
                    </div>
                    <div id="${uid}_modalButtons" class="modal-footer">
                    </div>
                </div>
            </div>`;
        document.body.appendChild(div);
    }
    if (noClose) { 
        document.getElementById(`${uid}_closeButton`).classList.add("d-none");
    }
    else {
        document.getElementById(`${uid}_closeButton`).classList.remove("d-none");
    }
    document.getElementById(`${uid}_modalTitle`).innerHTML = title;
    modalWindowContent = document.getElementById(`${uid}_modalContent`)
    modalWindowContent.innerHTML = "";
    modalWindowContent.appendChild(content);
    modalWindowButtons = document.getElementById(`${uid}_modalButtons`)
    modalWindowButtons.innerHTML = "";
    modalWindowButtons.appendChild(buttons);

    modalWindow = document.getElementById(`${uid}_modal`);
    $(modalWindow).modal({backdrop: 'static',keyboard: false});
    $(modalWindow).modal("show");
    modalWindow.getElementsByClassName("modal-dialog")[0].style.left = "0";
    modalWindow.getElementsByClassName("modal-dialog")[0].style.top = "0";
    $(".modal-header").on("mousedown", function(mousedownEvt) {
        var $draggable = $(this);
        var x = mousedownEvt.pageX - $draggable.offset().left,
            y = mousedownEvt.pageY - $draggable.offset().top;
        $("body").on("mousemove.draggable", function(mousemoveEvt) {
            $draggable.closest(".modal-dialog").offset({
                "left": mousemoveEvt.pageX - x,
                "top": mousemoveEvt.pageY - y
            });
        });
        $("body").one("mouseup", function() {
            $("body").off("mousemove.draggable");
        });
        $draggable.closest(".modal").one("bs.modal.hide", function() {
            $("body").off("mousemove.draggable");
        });
    });
    return modalWindow;
}

interfaceUtils.createDownloadDropdown = function(downloadRow, innerText, callback, comment, dropdownOptions) {
    var row = HTMLElementUtils.createRow(null);
    var selectDiv = document.createElement("div");
    var titleDiv = document.createElement("div");
    titleDiv.setAttribute("class", "col-12");
    titleDiv.innerHTML = `<b> ${innerText} </b>`
    row.appendChild(titleDiv);
    if (comment) {
        selectDiv.setAttribute("class", "col-6");
    }
    else {
        selectDiv.setAttribute("class", "col-12");
    }
    row.appendChild(selectDiv);
    random_select2_id = (Math.random() + 1).toString(36).substring(7);
    var paramSelect = {
        // eventListeners: {"change":callback},
        // "class": "btn btn-primary",
        // innerText: innerText
        class: "select2-select select2-select_" + random_select2_id
    }
    console.log("dropdownOptions", dropdownOptions);
    var DownloadDropdown = HTMLElementUtils.selectTypeDropDown(paramSelect);
    DownloadDropdown.setAttribute("data-placeholder", "Select from list (" + dropdownOptions.length + " items)")
    DownloadDropdown.style.width = "100%";
    selectDiv.appendChild(DownloadDropdown);
    if (comment) {
        var commentDiv = document.createElement("div");
        commentDiv.setAttribute("class", "col-6");
        if (comment)
            commentDiv.innerHTML = `<p style=" font-size:smaller; font-style: italic; color:#aaaaaa; padding-left:10px; margin-bottom: 0px;"> ${comment} </p>`
        row.appendChild(commentDiv);
    }

    downloadRow.appendChild(row);

    var timer = null;
    $(".select2-select_" + random_select2_id).select2({
        minimumResultsForSearch: 10,
        dropdownParent: selectDiv,
        ajax: {
            delay: 50,
            cache: true,
            transport: function(params, success, failure) {
                let pageSize = 100;
                console.log("params",params);
                let term = (params.data.term || '').toLowerCase();
                let page = (params.data.page || 1);
                
                if (timer)
                    clearTimeout(timer);

                timer = setTimeout(function(){
                    timer = null;
                    let results = dropdownOptions
                    .filter(function(f){
                        // your custom filtering here.
                        return f.text.toLowerCase().includes(term);
                    })

                    let paged = results.slice((page -1) * pageSize, page * pageSize);

                    let options = {
                        results: paged,
                        pagination: {
                            more: results.length >= page * pageSize
                        }
                    };
                    success(options);
                }, params.delay);
            }
        },
    })
    .on('select2:select', callback);
    return row;
}

interfaceUtils.createDownloadDropdownMarkers = function(options) {
    var downloadRow = document.getElementById("divMarkersDownloadButtons");
    interfaceUtils._mGenUIFuncs.generateUUID();
    if (!options.uid)
        options.uid=interfaceUtils._mGenUIFuncs.ctx.aUUID;
    var callback = function(e){
        params = e.params;
        if (e) {
            if ($('.select2-select').not(e.target)) {
                $('.select2-select').not(e.target).val(null).trigger('change');
            }
        }
        projectUtils.applySettings(options.settings);
        optionsCopy = JSON.parse(JSON.stringify(options));
        var dataURL = "";
        if (params.data.id === "") {return;}
        if (options.dropdownOptions) {
            dropdownOption = options.dropdownOptions[params.data.id];
            for (key in dropdownOption) {
                option_shifted = optionsCopy;
                var parameters = key.split(".");
                for (param_key in parameters) {
                    if (param_key == parameters.length-1) {
                        option_shifted[parameters[param_key]] = dropdownOption[key];
                    }
                    else {
                        option_shifted = option_shifted[parameters[param_key]]
                    }
                }
            }
        }
        else {
            //interfaceUtils._mGenUIFuncs.deleteTab(options.uid);
            dataURL = options.path[params.data.id];
            optionsCopy["path"] = dataURL;
        }
        interfaceUtils.generateDataTabUI(optionsCopy);
    }
    var dropdownOptions;
    dropdownOptions = [];
    if (options.dropdownOptions) {
        options.dropdownOptions.forEach (function (dropdownOption, index) {
            dropdownOptions.push({
                "id": index,
                "text": dropdownOption.optionName
            })
        });
    }
    else {
        options["path"].forEach (function (dataURL, index) {
            dropdownOptions.push({
                "id": index,
                "text": dataURL.split('/').reverse()[0].replace(/_/g, ' ').replace('.csv', '')
            })
        });
    }
    row = interfaceUtils.createDownloadDropdown(downloadRow, options.title, callback, options.comment, dropdownOptions);
    if (options.autoLoad) {
        if (options.autoLoad === true) {
            indexLoad = 0;
        }
        else {
            indexLoad = options.autoLoad;
        }
        $(row).find(".select2-select")
        .select2("trigger", "select", {
            data: { "id": dropdownOptions[indexLoad].id, "text":dropdownOptions[indexLoad].text }
        });
    }
}

interfaceUtils.createDownloadButton = function(downloadRow, innerText, callback, comment) {
    var row = HTMLElementUtils.createRow(null);
    var buttonDiv = document.createElement("div");
    buttonDiv.setAttribute("class", "col-6");
    row.appendChild(buttonDiv);
    var paramButton = {
        eventListeners: {"click":callback},
        "class": "btn btn-primary",
        innerText: innerText
    }
    var DownloadButton = HTMLElementUtils.createButton(paramButton);
    DownloadButton.style.width = "100%";
    buttonDiv.appendChild(DownloadButton);
    
    var commentDiv = document.createElement("div");
    commentDiv.setAttribute("class", "col-6");
    if (comment)
        commentDiv.innerHTML = `<p style=" font-size:smaller; font-style: italic; color:#aaaaaa; padding-left:10px; margin-bottom: 0px;"> ${comment} </p>`
    row.appendChild(commentDiv);

    downloadRow.appendChild(row);
    return row;
}

interfaceUtils.createDownloadButtonMarkers = function(options) {
    var downloadRow = document.getElementById("divMarkersDownloadButtons");
    interfaceUtils._mGenUIFuncs.generateUUID();
    if (!options.uid)
        options.uid=interfaceUtils._mGenUIFuncs.ctx.aUUID;
    var callback = function(e){
        if (e) {
            if ($('.select2-select').not(e.target)) {
                $('.select2-select').not(e.target).val(null).trigger('change');
            }
        };
        //interfaceUtils._mGenUIFuncs.deleteTab(options.uid);
        projectUtils.applySettings(options.settings);
        interfaceUtils.generateDataTabUI(options);
    }
    var buttonRow = interfaceUtils.createDownloadButton(downloadRow, options.title, callback, options.comment);
    if (options.autoLoad) {
        setTimeout(function(){callback(null)},500);
        buttonRow.style.display="none";
    }
}

interfaceUtils.createDownloadDropdownRegions = function(options) {
    var downloadRow = document.getElementById("divRegionsDownloadButtons");
    var callback = function(e){
        params = e.params
        projectUtils.applySettings(options.settings);
        var dataURL = params.data.id;
        if (dataURL == "") return;
        regionUtils.JSONToRegions(dataURL)
    }
    var dropdownOptions;
    if (options.autoLoad) {
        dropdownOptions = [];
    }
    else {
        dropdownOptions = [];
    }
    options["path"].forEach (function (dataURL) {
        dropdownOptions.push({
            "id": dataURL,
            "text": dataURL.split('/').reverse()[0].replace(/_/g, '').replace('.json', '')
        })
    });
    interfaceUtils.createDownloadDropdown(downloadRow, options.title, callback, options.comment, dropdownOptions);
    //var label = document.getElementById("label_ISS_csv");
    if (options.autoLoad) {
        setTimeout(function(){callback(null, {'selected':options["path"][0]})},500);
    }
    //else { label.innerHTML = "Or import gene expression from CSV file:"; }
}

interfaceUtils.createDownloadButtonRegions = function(options) {
    var downloadRow = document.getElementById("divRegionsDownloadButtons");
    var callback = function(e){
        projectUtils.applySettings(options.settings);
        regionUtils.JSONToRegions(options.path)
    }
    var buttonRow = interfaceUtils.createDownloadButton(downloadRow, options.title, callback, options.comment);
    if (options.autoLoad) {
        setTimeout(function(){callback(null)},500);
        buttonRow.style.display="none";
    }
}

interfaceUtils.addMenuItem = function(itemTree, callback, before) {
    itemID = "menubar";
    rootElement = document.querySelector("#navbar-menu .navbar-nav");
    for (var i = 0; i<itemTree.length; i++) {
        itemID += "_" + HTMLElementUtils.stringToId(itemTree[i]);
        if (!document.getElementById(itemID)) {
            liItem = HTMLElementUtils.createElement({"kind":"li", "extraAttributes":{"class":"nav-item dropdown"}})
            if (i == 0)
                rootElement.insertBefore(liItem, document.getElementById("nav-item-title"));
            else if (before)
                rootElement.prepend(liItem);
            else
                rootElement.append(liItem);
            
            if (i == 0 && i == itemTree.length -1) {
                aElement = HTMLElementUtils.createElement({"kind":"a", "id":"a_"+itemID, "extraAttributes":{"class":"nav-link active","href":"#"}})
                liItem.appendChild(aElement);
                aElement.addEventListener("click",function (event) {
                    callback();
                });
                spanMore = "";
            }
            else if (i == 0) {
                aElement = HTMLElementUtils.createElement({"kind":"a", "id":"a_"+itemID, "extraAttributes":{"class":"nav-link dropdown-toggle active","href":"#", "data-bs-toggle":"dropdown", "aria-haspopup":"true", "aria-expanded":"false"}})
                liItem.appendChild(aElement);
                ulItem = HTMLElementUtils.createElement({"kind":"ul", "id":itemID, "extraAttributes":{"class":"dropdown-menu dropdown-submenu"}})
                liItem.appendChild(ulItem);
                rootElement = ulItem;
                spanMore = "";
            }
            else if (i != itemTree.length -1) {
                aElement = HTMLElementUtils.createElement({"kind":"a", "id":"a_"+itemID, "extraAttributes":{"class":"dropdown-item","href":"#", "data-bs-toggle":"dropdown", "aria-haspopup":"true", "aria-expanded":"false"}})
                liItem.appendChild(aElement);
                ulItem = HTMLElementUtils.createElement({"kind":"ul", "id":itemID, "extraAttributes":{"class":"dropdown-menu dropdown-submenu"}})
                liItem.appendChild(ulItem);
                rootElement = ulItem;
                spanMore = " &raquo;";
            }
            else {
                aElement = HTMLElementUtils.createElement({"kind":"a", "id":"a_"+itemID, "extraAttributes":{"class":"dropdown-item", "href":"#"}})
                liItem.appendChild(aElement);
                aElement.addEventListener("click",function (event) {
                    callback();
                });
                spanMore = "";
            }
            var spanElement = document.createElement("span");
            aElement.appendChild(spanElement);
            spanElement.innerHTML = itemTree[i] + spanMore;
        }
        else {
            rootElement = document.getElementById(itemID);
        }
    }

    /////// Prevent closing from click inside dropdown
    document.querySelectorAll('.dropdown-menu').forEach(function(element){
        element.addEventListener('click', function (e) {
            e.stopPropagation();
        });
    })
}

interfaceUtils.addPluginAccordion = function (pluginID, pluginName) {
    var pluginID = HTMLElementUtils.stringToId(pluginID);
    var pluginsAccordions = document.getElementById("pluginsAccordions");
    var accordion_item = document.getElementById("PluginAccordionItem-" + pluginID);
    if (!accordion_item) {
        var accordion_item = HTMLElementUtils.createElement({
            kind: "div",
            extraAttributes: {
                class: "accordion-item plugin-accordion",
                id: "PluginAccordionItem-" + pluginID
            }
        });
        pluginsAccordions.appendChild(accordion_item);
        var accordion_header = HTMLElementUtils.createElement({
            kind: "h2",
            extraAttributes: {
                class: "accordion-header",
                id: "pluginHeading-" + pluginID
            }
        });
        accordion_item.appendChild(accordion_header);
        var accordion_header_button = HTMLElementUtils.createElement({
            kind: "button",
            innerHTML: pluginName,
            extraAttributes: {
                "type": "button",
                "class": "accordion-button collapsed",
                "id": "pluginHeading-" + pluginID,
                "data-bs-toggle": "collapse",
                "data-bs-target": "#" + "plugin-" + pluginID,
                "aria-expanded": "true",
                "aria-controls": "collapseOne"
            }
        });
        accordion_header.appendChild(accordion_header_button);
        
        var accordion_content = HTMLElementUtils.createElement({
            kind: "div",
            extraAttributes: {
                class: "accordion-collapse collapse px-2",
                id: "plugin-" + pluginID,
                "aria-labelledby":"headingOne",
                "data-bs-parent":"#pluginsAccordions"
            }
        });
        accordion_item.appendChild(accordion_content);
        accordion_content.innerHTML = "[Plugin loading...]"
    }
    $('#title-tab-plugins').tab('show');
    $('#' + "plugin-" + pluginID).collapse('show', {parent: pluginsAccordions});
    document.getElementById("title-tab-plugins").classList.remove("d-none");
    return document.getElementById("plugin-" + pluginID);
}
