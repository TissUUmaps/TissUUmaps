/**
 * @file filterUtils.js Interface to ask information to OSD
 * @author Christophe Avenel
 * @see {@link filterUtils}
 */

/**
 * @namespace filterUtils
 * @property {Bool}   filterUtils._filtersUsed - 
 * @property {Object} filterUtils._filters - 
 * @property {Object} filterUtils._filterItems - 
 */
 filterUtils = {
    // Choose between ["Brightness", "Exposure", "Hue", "Contrast", "Vibrance", "Noise", 
    //                 "Saturation","Gamma","Invert","Greyscale","Threshold","Erosion","Dilation"]
    _filtersUsed: ["Saturation","Brightness"],
    _filters: {
        "Brightness":{
            params:{
                type:"range",
                min:-100,
                max:100,
                step:1,
                value:0
            },
            filterFunction: function (value) {
                if (value == 0) {  return function (context, callback) {callback();}}
                return function(context, callback) {
                    Caman(context.canvas, function() {
                        this.brightness(value);
                        this.render(callback);
                    })
                }
            }
        },
        "Exposure":{
            params:{
                type:"range",
                min:-100,
                max:100,
                step:1,
                value:0
            },
            filterFunction: function (value) {
                if (value == 0) {  return function (context, callback) {callback();}}
                return function(context, callback) {
                    Caman(context.canvas, function() {
                        this.exposure(value);
                        this.render(callback);
                    })
                }
            }
        },
        "Hue":{
            params:{
                type:"range",
                min:0,
                max:100,
                step:1,
                value:0
            },
            filterFunction: function (value) {
                if (value == 0) {  return function (context, callback) {callback();}}
                return function(context, callback) {
                    Caman(context.canvas, function() {
                        this.hue(value);
                        this.render(callback);
                    })
                }
            }
        },
        "Contrast":{
            params:{
                type:"range",
                min:0,
                max:3,
                step:0.01,
                value:1
            },
            filterFunction: function (value) {
                if (value == 1) {  return function (context, callback) {callback();}}
                return OpenSeadragon.Filters.CONTRAST(value);
            }
        },
        "Vibrance":{
            params:{
                type:"range",
                min:-250,
                max:250,
                step:10,
                value:0
            },
            filterFunction: function (value) {
                if (value == 0) {  return function (context, callback) {callback();}}
                return function(context, callback) {
                    Caman(context.canvas, function() {
                        this.vibrance(value);
                        this.render(callback);
                    })
                }
            }
        },
        "Noise":{
            params:{
                type:"range",
                min:0,
                max:100,
                step:10,
                value:0
            },
            filterFunction: function (value) {
                if (value == 0) {  return function (context, callback) {callback();}}
                return function(context, callback) {
                    Caman(context.canvas, function() {
                        this.noise(value);
                        this.render(callback);
                    })
                }
            }
        },
        "Saturation":{
            params:{
                type:"range",
                min:-100,
                max:100,
                step:10,
                value:0
            },
            filterFunction: function (value) {
                if (value == 0) {  return function (context, callback) {callback();}}
                return function(context, callback) {
                    Caman(context.canvas, function() {
                        this.saturation(value);
                        this.render(callback);
                    })
                }
            }
        },
        "Gamma":{
            params:{
                type:"range",
                min:-0,
                max:2,
                step:0.1,
                value:1
            },
            filterFunction: function (value) {
                if (value == 1) {  return function (context, callback) {callback();}}
                return function(context, callback) {
                    Caman(context.canvas, function() {
                        this.gamma(value);
                        this.render(callback);
                    })
                }
            }
        },
        "Invert":{
            params:{
                type:"checkbox"
            },
            filterFunction: function () {
                return OpenSeadragon.Filters.INVERT();
            }
        },
        "Greyscale":{
            params:{
                type:"checkbox"
            },
            filterFunction: function () {
                return OpenSeadragon.Filters.GREYSCALE();
            }
        },
        "Threshold":{
            params:{
                type:"range",
                min:0,
                max:256,
                step:1,
                value:256
            },
            filterFunction: function (value) {
                if (value == 256) {  return function (context, callback) {callback();}}
                return OpenSeadragon.Filters.THRESHOLDING(value);
            }
        },
        "Erosion":{
            params:{
                type:"range",
                min:1,
                max:11,
                step:2,
                value:1
            },
            filterFunction: function (value) {
                if (value == 1) {  return function (context, callback) {callback();}}
                return OpenSeadragon.Filters.MORPHOLOGICAL_OPERATION(value, Math.min);
            }
        },
        "Dilation":{
            params:{
                type:"range",
                min:1,
                max:11,
                step:2,
                value:1
            },
            filterFunction: function (value) {
                if (value == 1) {  return function (context, callback) {callback();}}
                return OpenSeadragon.Filters.MORPHOLOGICAL_OPERATION(value, Math.max);
            }
        }
    },
    _filterItems:{}
}

/** 
 * @param {Number} filterName 
 * Get params for a given filter */
filterUtils.getFilterParams = function(filterName) {
    filterParams = filterUtils._filters[filterName].params;
    filterParams.eventListeners = {
        "change": filterUtils.getFilterItems
    };
    filterParams["class"] = "filterInput";
    filterParams.filter = filterName;
    return filterParams;
}

/** 
 * @param {Number} filterName Index of an OSD tile source
 * Get function for a given filter */
filterUtils.getFilterFunction = function(filterName) {
    return filterUtils._filters[filterName].filterFunction;
}

/** 
 * apply all filters to all layers
 *  */
 filterUtils.applyFilterItems = function(calledItems) {
    var caman = Caman;
	caman.Store.put = function() {};

    var op = tmapp["object_prefix"];
    if (!overlayUtils.areAllFullyLoaded() || !tmapp[op + "_viewer"].world.getItemAt(Object.keys(filterUtils._filterItems).length-1)) {
        setTimeout(function() {
            if (calledItems == filterUtils._filterItems)
                filterUtils.applyFilterItems(calledItems);
        }, 100);
        return;
    }
    filters = [];
    for (const layer in filterUtils._filterItems) {
        processors = [];
        for(var filterIndex=0;filterIndex<filterUtils._filterItems[layer].length;filterIndex++) {
            processors.push(
                filterUtils._filterItems[layer][filterIndex].filterFunction(filterUtils._filterItems[layer][filterIndex].value)
            );
        }
        filters.push({
            items: tmapp[op + "_viewer"].world.getItemAt(layer),
            processors: processors
        });
    };
    tmapp[op + "_viewer"].setFilterOptions({
        filters: filters
    });
}

/** 
 * Get filter functions and values from html ranges and checkboxes
 *  */
filterUtils.getFilterItems = function() {
    filterInputsRanges = document.getElementsByClassName("filterInput");
    items = {}
    for (i = 0; i < filterInputsRanges.length; i++) {
        filterInputsRanges[i];
        if (!items[filterInputsRanges[i].getAttribute("layer")]) {
            items[filterInputsRanges[i].getAttribute("layer")] = []
        }
        filterFunction = filterUtils.getFilterFunction(filterInputsRanges[i].getAttribute("filter"));
        if (filterInputsRanges[i].type == "range")
            inputValue = filterInputsRanges[i].value;
        else
            inputValue = filterInputsRanges[i].checked;
        if (inputValue) {
            items[filterInputsRanges[i].getAttribute("layer")].push(
                {
                    filterFunction: filterUtils.getFilterFunction(filterInputsRanges[i].getAttribute("filter")),
                    value: inputValue
                }
            );
        }
    }
    filterUtils._filterItems = items;
    filterUtils.applyFilterItems(items);
}
