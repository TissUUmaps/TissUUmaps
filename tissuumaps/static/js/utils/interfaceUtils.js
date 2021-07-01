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
* Listen to an event of an element, if the element doesnçt exist get a warning. */
interfaceUtils.listen= function(domid,event,handler,debug){
    var dbg=debug || false;
    //console.log(dbg)
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

	if (!dataUtils[op + "_barcodeGarden"]) {
		alert("Load markers first");
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
 *  Get the info of the region that has to be changed */
interfaceUtils.changeRegionUI = function (callingbutton) {
	var regionid = callingbutton[0].getAttribute("parentRegion");
	regionUtils.changeRegion(regionid);
}

/** 
* @param {String} domid The id of the select element
* @param {String[]} Array of strings containing elements to add to the select
* Add options to a select element */
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
* Add options to a select element using Objects with the keys: "value* and "innerHTML" */
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
* Erase all options in a select element */
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
* Make an element invisible */
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
* Make an element visible */
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
* Disable an element */
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
* Enable an element */
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
* Ask if an element is enabled */
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
* Get the selected option in a sleect element */
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
* Call the main dom.getElementsByClassName function with a warning if no elements exist */
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
* Call the main dom.getElementsByTagName function with a warning if no elements exist */
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
* Get the an element and warn if none exists */
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
* Get the an element and warn if none exists */
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
* @return {HTMLelement | null} HTMl element
* Get the an element and warn if none exists */
interfaceUtils.getElementById=function(domid){
    var elem= document.getElementById(domid);
    if(elem){
        return elem;
    }else{
        console.log("Element with id "+domid+" doesn't exist");
        return null;
    }
}

/** 
* @param {String} domid The id of the element
* @return {String | null} HTMl element
* Get the value of a dom element and warn if element does not exist*/
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
* Get the innerHTML of a dom element and warn if element does not exist*/
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
* associated components. For a single viewer the default is "ISS", 
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
 * find and actiate main tabs */
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
 * hides all the tabs that should not he  displayed except a itself */
interfaceUtils.hideTabsExcept = function (a) {
    //get a tag, get it's closes ul check the level, deactivate all but this
    const regex1 = RegExp("L([0-9]+)-tabs", 'g');
    //first, get closest ul contaninig list of a links
    var closestul = a.closest("ul");
    var level = 0;

    //find main child tabs and activate them
    
    //find href to know which id to look for and which to hide
    var elid = a[0].href.split("#")[1]
    console.log(elid,":")
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
        console.log(uls[i])
        var ulsas = uls[i].getElementsByTagName("a");
        for (var j = 0; j < ulsas.length; j++) {
            ana=ulsas[j].href.split("#")[1];
            //console.log(ana)
            //console.log("!ana.includes(elid)", !ana.includes(elid))
            if(!ana.includes(elid)){
                //only turn non elids
                as.push(ana)
                ulsas[j].classList.remove("active")
            }
        }    
    }

    for(var i=0;i<as.length;i++){
        //find elements with this id and deactivate them
        //console.log(as[i]);
        var el=document.getElementById(as[i]);
        
        if(el!==null && el.classList.length>0){
            el.classList.remove("active");
            el.classList.remove("show");
        }
    }
   
}

/** 
 * @param {object} a dom object of the a tag 
 * hides all the tabs that should not he  displayed except a itself */
 interfaceUtils.toggeRightPanel = function (a) {
    var op = tmapp["object_prefix"];
    var menu=document.getElementById(op + "_menu");
    var main=document.getElementById(op + "_viewer_container");
    var btn=document.getElementById(op + "_collapse_btn");
    var style = window.getComputedStyle(menu);
    if (style.display === 'none') {
        menu.style.display = "block";
        main.style.width = "66.66666%";
        main.style.maxWidth = "Calc(100% - 506px)";
        btn.innerText = ">";
    }
    else {
        menu.style.display = "none";
        main.style.width = "100%";
        main.style.maxWidth = "";
        btn.innerText = "<";
    }
}
