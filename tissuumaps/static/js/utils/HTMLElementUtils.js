/**
* @file HTMLElementUtils.js Wrappers to common dom element creation 
* with options to make the creation of an HTML element shorter
* @author Leslie Solorzano
* @see {@link HTMLElementUtils}
*/
/**
* @namespace HTMLElementUtils
*/
HTMLElementUtils = {}

/** Create a checkbox input  */
HTMLElementUtils.inputTypeCheckbox = function (params) {
    if (!params) {
        var checkbox = document.createElement("input");
        checkbox.setAttribute("type", "checkbox");
        return checkbox;
    }
    var checkbox = document.createElement("input");
    checkbox.setAttribute("type", "checkbox");
    (params.id || null ? checkbox.setAttribute("id", params.id) : null);
    (params["class"] || null ? checkbox.setAttribute("class", params["class"]) : null);
    (params["checked"] || null ? checkbox.checked=params["checked"] : null);
    var extraAttributes = params.extraAttributes || null;
    if (extraAttributes) {
        for (var attr in extraAttributes) {
            checkbox.setAttribute(attr, extraAttributes[attr]);
        }
    }
    var eventListeners = params.eventListeners || null;
    if (eventListeners) {
        for (var message in eventListeners) {
            checkbox.addEventListener(message, eventListeners[message]);
        }
    }
    return checkbox;
}

/** Create a color input  */
HTMLElementUtils.inputTypeColor = function (params) {
    if (!params) {
        var color = document.createElement("input");
        color.setAttribute("type", "color");
        return color;
    }
    var color = document.createElement("input");
    color.setAttribute("type", "color");
    (params.id || null ? color.setAttribute("id", params.id) : null);
    (params["class"] || null ? color.setAttribute("class", params["class"]) : null);
    var extraAttributes = params.extraAttributes || null;
    if (extraAttributes) {
        for (var attr in extraAttributes) {
            color.setAttribute(attr, extraAttributes[attr]);
        }
    }
    var eventListeners = params.eventListeners || null;
    if (eventListeners) {
        for (var message in eventListeners) {
            color.addEventListener(message, eventListeners[message]);
        }
    }
    return color;
}

/** Create a text input  */
HTMLElementUtils.inputTypeText = function (params) {
    if (!params) {
        var text = document.createElement("input");
        text.setAttribute("type", "text");
        return text;
    }
    var text = document.createElement("input");
    text.setAttribute("type", "text");
    (params.id || null ? text.setAttribute("id", params.id) : null);
    (params["class"] || null ? text.setAttribute("class", params["class"]) : null);
    var extraAttributes = params.extraAttributes || null;
    if (extraAttributes) {
        for (var attr in extraAttributes) {
            text.setAttribute(attr, extraAttributes[attr]);
        }
    }
    return text;
}

/** Create a range input  */
HTMLElementUtils.inputTypeRange = function (params) {
    if (!params) {
        var rangeInput = document.createElement("input");
        rangeInput.type = "range";
        return rangeInput
    }
    var rangeInput = document.createElement("input");
    rangeInput.type = "range";
    (params.id || null ? rangeInput.setAttribute("id", params.id) : null);
    (params["class"] || null ? rangeInput.setAttribute("class", params["class"]) : null);
    (params.min || null ? rangeInput.setAttribute("min", params.min) : null);
    (params.max || null ? rangeInput.setAttribute("max", params.max) : null);
    (params.step || null ? rangeInput.setAttribute("step", params.step) : null);
    var eventListeners = params.eventListeners || null;
    if (eventListeners) {
        for (var message in eventListeners) {
            rangeInput.addEventListener(message, eventListeners[message]);
        }
    }
    return rangeInput;
}


/** Create a drop down select  */
HTMLElementUtils.selectTypeDropDown = function (params) {
    if (!params) {
        return document.createElement("select");
    }
    var select = document.createElement("select");
    (params.id || null ? select.setAttribute("id", params.id) : null);
    (params["class"] || null ? select.setAttribute("class", params["class"]) : null);
    var options = params.options || null;
    if (options) {
        options.forEach(function (symbol, i) {
            var option = document.createElement("option");
            if (symbol.text) {
                option.value = symbol.value;
                option.text = symbol.text;
            }
            else {
                option.value = i;
                option.text = symbol;
            }
            select.appendChild(option);
        });
    }
    var eventListeners = params.eventListeners || null;
    if (params.eventListeners) {
        for (var message in eventListeners) {
            select.addEventListener(message, eventListeners[message]);
        }
    }
    if (params.extraAttributes) {
        for (var attr in params.extraAttributes) {
            select.setAttribute(attr, params.extraAttributes[attr]);
        }
    }
    return select;
}

/** Create an HTML element with the common tags (e.g a,p,h1) */
HTMLElementUtils.createElement = function (params) {
    if (!params) {
        return null;
    }
    var id = params.id || null;
    var kind = params.kind || null;
    var innerText = params.innerText || null;
    var innerHTML = params.innerHTML || null;

    var element = (kind ? document.createElement(kind) : null);
    if (!element) return null;
    (id ? element.setAttribute("id", id) : null);
    (innerText ? element.appendChild(document.createTextNode(innerText)) : null);
    (innerHTML ? element.innerHTML = innerHTML : null);
    if (params.extraAttributes) {
        for (var attr in params.extraAttributes) {
            element.setAttribute(attr, params.extraAttributes[attr]);
        }
    }
    return element;
}

/** Create a booststrap panel */
HTMLElementUtils.createPanel = function (params) {
    if (!params) {
        var panelClass = "panel-default";
        var panel = document.createElement("div");
        panel.setAttribute("class", "panel " + panelClass);
        return panel;
    }
    var panelClass = params.panelClass || "panel-default";
    var panel = document.createElement("div");
    panel.setAttribute("class", "panel " + panelClass);
    (params.id || null ? panel.setAttribute("id", params.id) : null);

    var panel_heading = document.createElement("div");
    panel_heading.setAttribute("class", "panel-heading");
    (params.headingInnerText || null ? panel_heading.appendChild(document.createTextNode(params.headingInnerText)) : null);

    var panel_body = document.createElement("div");
    panel_body.setAttribute("class", "panel-body");
    panel.appendChild(panel_heading);
    panel.appendChild(panel_body);
    return panel;
}

/** Create a booststrap row */
HTMLElementUtils.createRow = function (params) {
    if (!params) {
        var row = HTMLElementUtils.createElement({ kind: "div" });
        row.setAttribute("class", "row py-1");
        return row;
    }
    var row = HTMLElementUtils.createElement({ kind: "div" });
    (params.id || null ? row.setAttribute("id", params.id) : null);
    row.setAttribute("class", "row py-1");
    if (params.divisions) {
        params.divisions.forEach(function (division) {
            row.appendChild(HTMLElementUtils.createColumn(division));
        });
    }
    if (params.extraAttributes) {
        for (var attr in extraAttributes) {
            row.setAttribute(attr, extraAttributes[attr]);
        }
    }
    return row;

}

/** Create a booststrap column */ 
HTMLElementUtils.createColumn = function (params) {
    if (!params) {
        var width = 2;
        var column = HTMLElementUtils.createElement({ kind: "div" });
        column.setAttribute("class", "col-xs-" + width + " col-sm-" + width + " col-md-" + width + " col-lg-" + width);
        return column;
    }
    var width = (params.width || null ? params.width : 2);
    var column = HTMLElementUtils.createElement({ kind: "div" });
    (params.id || null ? column.setAttribute("id", params.id) : null);
    if (width == 0) {
        column.setAttribute("class", "col");
    } else {
        column.setAttribute("class", "col-xs-" + width + " col-sm-" + width + " col-md-" + width + " col-lg-" + width);
    }
    if (params.extraAttributes) {
        for (var attr in params.extraAttributes) {
            if (attr == "class")
                column.setAttribute('class', column.getAttribute('class') + ' ' + params.extraAttributes[attr]);
            else
                column.setAttribute(attr, params.extraAttributes[attr]);
        }
    }
    return column;
}

/** Create a button */
HTMLElementUtils.createButton = function (params) {
    if (!params) {
        var button = document.createElement("button");
        button.innerHTML = "unnamed";
        return button;
    }
    var button = document.createElement("button");
    (params.id || null ? button.setAttribute("id", params.id) : null);
    (params.innerText || null ? button.innerHTML = params.innerText : null);
    (params["class"] || null ? button.setAttribute("class", params["class"]) : null);
    var eventListeners = params.eventListeners || null;
    if (eventListeners) {
        for (var message in eventListeners) {
            button.addEventListener(message, eventListeners[message]);
        }
    }
    if (params.extraAttributes) {
        for (var attr in params.extraAttributes) {
            button.setAttribute(attr, params.extraAttributes[attr]);
        }
    }
    return button;
}

/** Create a table */
HTMLElementUtils.createTable = function (params) {
    if (!params) {
        return document.createElement("table");
    }
    var table = document.createElement("table");
    (params.id || null ? table.setAttribute("id", params.id) : null);
    (params.tableClass || null ? table.setAttribute("class", params.tableClass) : null);

    if (params.extraAttributes) {
        for (var attr in params.extraAttributes) {
            table.setAttribute(attr, params.extraAttributes[attr]);
        }
    }
    return table;

}

/** Create a form */
HTMLElementUtils.createForm = function (params) {
    if (!params) {
        var form = document.createElement("form");
        return form;
    }
    var form = document.createElement("form");
    if (params.extraAttributes) {
        for (var attr in params.extraAttributes) {
            form.setAttribute(attr, params.extraAttributes[attr]);
        }
    }
    return form;
}

/** Create a color in YCbCr space to divide between the possible 4 letters */
HTMLElementUtils.determinsticHTMLColor = function (key) {
    //A Red, T Green, C Bluemagenta, G yellow
    var red = 0; var green = 0; var blue = 0;
    var U = 0; var V = 0; var y = 128;

    if (!isNaN(parseInt(key))) {
        key = parseInt(key,10).toString(4).split("").reverse().join("");
    }
    
    var maincolor = key.charAt(0).toLowerCase();

    ggroup = ['g', 'i', 's', 'd', 'w', 'z','1','5','9'];
    agroup = ['a', 'e', 'o', 'l', 'p', 'b', 'k', 'j','2','6'];
    cgroup = ['c', 'm', 'f', 'v','3','7','q'];
    tugroup = ['t', 'r', 'n', 'u', 'h', 'y', 'x','4','8','0' ];

    if (ggroup.includes(maincolor)) { U = 0; V = 0; } 
    else if (tugroup.includes(maincolor)) { U = 0; V = 255; }
    else if (cgroup.includes(maincolor)) { U = 255; V = 0; }
    else if (agroup.includes(maincolor)) { U = 255; V = 255; } 
    else{ U = 0; V = 255; }

    var second = key.charAt(1).toLowerCase();

    if (agroup.includes(second)) { U += 80; V += 80; } if (cgroup.includes(second)) { U += 80; V += -80; }
    if (ggroup.includes(second)) { U += -80; V += -80; } if (tugroup.includes(second)) { U += -80; V += 80; }

    if (U > 255) U = 255; if (V > 255) V = 255;
    if (U < 0) U = 0; if (V < 0) V = 0;

    var third = key.charAt(2).toLowerCase();

    if (agroup.includes(third)) { y += 35; } if (cgroup.includes(third)) { y += 35; }
    if (ggroup.includes(third)) { y -= 35; } if (tugroup.includes(third)) { y -= 35; }

    red = Math.floor(y + 1.402 * (V - 128));
    green = Math.floor(y - 0.344136 * (U - 128) - 0.714136 * (V - 128));
    blue = Math.floor(y + 1.772 * (U - 128));

    if (red >= 255) red -= red % 255; if (green >= 255) green -= green % 255; if (blue >= 255) blue -= blue % 255;

    if (red < 0) red = 0; if (green < 0) green = 0; if (blue < 0) blue = 0;

    var reds = red.toString(16).toUpperCase();
    var greens = green.toString(16).toUpperCase();
    var blues = blue.toString(16).toUpperCase();

    if (reds.length == 1) reds = "0" + reds; if (greens.length == 1) greens = "0" + greens; if (blues.length == 1) blues = "0" + blues;

    return "#" + reds + greens + blues;

}

/** 
* getFirstChildByClass  */
HTMLElementUtils.getFirstChildByClass = function (e, c) {
    var thisChild = null;
    e.childNodes.forEach(function (child) {
        var childClasses = child.className.split(" ");
        for (var i in childClasses) {
            //console.log(childClasses[i]);
            if (c == childClasses[i]) {
                thisChild = child;
                break;
            }
        }
    });
    return thisChild;
}

/** Create an id from any string */
HTMLElementUtils.stringToId = function (inputString) {
    if (!inputString) return "";
    return inputString.replace(/\W/g, '')
}
