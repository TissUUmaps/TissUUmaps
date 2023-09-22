/**
 * @file projectUtils.js
 * @author Christophe Avenel
 * @see {@link projectUtils}
 */

/**
 * @namespace projectUtils
 * @version projectUtils 2.0
 * @classdesc The root namespace for projectUtils.
 */
 pluginUtils = {
    _pluginList: []
 }

pluginUtils.addPlugin = function (pluginID) {
    if (pluginUtils._pluginList.includes(pluginID)) {
        return;
    }
    pluginUtils._pluginList.push(pluginID);
    interfaceUtils.addMenuItem(["Plugins",pluginID],function() {
        pluginUtils.startPlugin(pluginID);
    });
}

pluginUtils.startPlugin = function (pluginID, options) {
    var script = document.createElement('script');
    script.src = `plugins/${pluginID}.js?date=${Date.now()}`;
    document.head.appendChild(script);
    script.onload = function () {
        if (!window[pluginID]) {
            interfaceUtils.alert(`Impossible to load plugins/${pluginID}.js file.`);return
        }
        pluginTitle = window[pluginID]["name"];
        if (!pluginTitle)
            pluginTitle = pluginID;
        
        pluginDiv = interfaceUtils.addPluginAccordion(pluginID, pluginTitle);
        if (window[pluginID].parameters !== undefined) {
            pluginUtils.loadParameters(pluginID, pluginDiv, window[pluginID].parameters);
        }

        if (options !== undefined) {
            for (var option of options) {
                window[pluginID].set(option.name, option.value);
            }
        }

        window[pluginID]["init"](pluginDiv);
    }
}

pluginUtils.loadParameters = function (pluginID, pluginDiv, parameters) {
    window[pluginID].set = function(paramName, value) {
        window[pluginID][paramName] = value;
        var parameterID = pluginID + "_" + paramName;
        if (document.getElementById(parameterID) !== null) {
            document.getElementById(parameterID).value = value;
            document.getElementById(parameterID).checked = value;
        }
    }
    window[pluginID].get = function(paramName) {
        return window[pluginID][paramName];
    }
    window[pluginID].getInputID = function(paramName) {
        var parameterID = pluginID + "_" + paramName;
        return parameterID;
    }
    window[pluginID].api = function(endpoint, data, success, error) {
        $.ajax({
            // Post select to url.
            type: "post",
            url: "/plugins/" + pluginID + "/" + endpoint,
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify(data),
            success: function (data) {
              success(data);
            },
            complete: function (data) {
              // do something, not critical.
            },
            error: (error) ? error : function (data) {
              interfaceUtils.alert(data.responseText.replace("\n","<br/>"),"Error on the plugin's server response:");
            },
          });
    }
    
    pluginDiv.innerHTML = "";
    let activeDiv = pluginDiv;
    for (var parameterName in parameters) {
        let parameter = parameters[parameterName];
        if (parameter.label === undefined) {
            parameter.label = parameterName;
        }
        var parameterID = pluginID + "_" + parameterName;
        var row = HTMLElementUtils.createRow({});
        var col1 = HTMLElementUtils.createColumn({ width: 12 });
        if (parameter.type == "section") {
            let row0=HTMLElementUtils.createElement({"kind":"h6", "extraAttributes":{"class":""}});
            row0.innerText = parameter.title;
            row0.style.borderBottom = "1px solid #aaa";
            row0.style.padding = "3px";
            row0.style.marginTop = "8px";

            let row01=HTMLElementUtils.createElement({"kind":"p", "extraAttributes":{"class":""}});
            row01.innerHTML = "<i>"+parameter.label+"</i>";
            console.log(parameter, parameter.collapsed);
            row0?.setAttribute("data-bs-toggle", "collapse");
            row0?.setAttribute("data-bs-target", "#collapsedSection_"+parameterID);
            row0?.setAttribute("aria-controls", "collapse_advanced");
            row0?.setAttribute("style", "cursor: pointer;");
            row0?.setAttribute("title", "Click to expand");
            let newDiv = document.createElement("div");
            newDiv.setAttribute("id", "collapsedSection_"+parameterID);
            row.appendChild(col1);
            col1.appendChild(row0);
            newDiv.appendChild(row01);
            
            if (parameter.collapsed) {
                row0?.setAttribute("aria-expanded", "false");
                newDiv.setAttribute("class", "collapse");
                row0?.setAttribute("class", "collapse_button_transform border-bottom-0 p-1 collapsed");
            }
            else {
                row0?.setAttribute("aria-expanded", "true");
                newDiv.setAttribute("class", "collapse show");
                row0?.setAttribute("class", "collapse_button_transform border-bottom-0 p-1");
            }
            

            pluginDiv.appendChild(row);
            pluginDiv.appendChild(newDiv);
            activeDiv = newDiv;
            continue;
        }
        else if (parameter.type == "label") {
            row0=HTMLElementUtils.createElement({
                id: parameterID,
                "kind":"p",
                "extraAttributes":{"class":""}
            });
            row0.innerHTML = parameter.label;
            
            row.appendChild(col1);
            col1.appendChild(row0);
            
            activeDiv.appendChild(row);
            continue;
        }
        else if (parameter.type == "button") {
            button11 = HTMLElementUtils.createButton({
                id: parameterID,
                extraAttributes: {
                    class: "btn btn-secondary btn-sm",
                    data_parameterName: parameterName
                },
            });
            button11.innerHTML = parameter.label;

            button11.addEventListener("click", (event) => {
                var parameterName = event.target.getAttribute("data_parameterName");
                if (window[pluginID].inputTrigger !== undefined) {
                    window[pluginID].inputTrigger(parameterName);
                }
            });
            col1.appendChild(button11);
            row.appendChild(col1);
            activeDiv.appendChild(row);
        }
        else if (parameter.type == "number" || parameter.type == "text") {
            var extraAttributes = {
                class: "form-text-input form-control",
                type: parameter.type,
                value: window[pluginID].get(parameterName),
                data_parameterName: parameterName
            }
            if (parameter.attributes !== undefined) {
                extraAttributes = {...extraAttributes, ...parameter.attributes};
            }
            var input11 = HTMLElementUtils.createElement({
                kind: "input",
                id: parameterID,
                extraAttributes: extraAttributes,
            });
            var label11 = HTMLElementUtils.createElement({
                kind: "label",
                extraAttributes: { for: parameterID },
            });
            label11.innerHTML = parameter.label;
            col1.appendChild(label11);
            col1.appendChild(input11);
            row.appendChild(col1);
            activeDiv.appendChild(row);
        }
        else if (parameter.type == "select") {
            var extraAttributes = {
                class: "form-select form-select-sm",
                value: window[pluginID].get(parameterName),
                data_parameterName: parameterName
            }
            if (parameter.attributes !== undefined) {
                extraAttributes = {...extraAttributes, ...parameter.attributes};
            }
            var input11 = HTMLElementUtils.createElement({
                kind: "select",
                id: parameterID,
                extraAttributes: extraAttributes,
            });
            var label11 = HTMLElementUtils.createElement({
                kind: "label",
                extraAttributes: { for: parameterID },
            });
            label11.innerHTML = parameter.label;
            col1.appendChild(label11);
            col1.appendChild(input11);
            row.appendChild(col1);
            activeDiv.appendChild(row);

            if (parameter.options !== undefined) {
                interfaceUtils.addElementsToSelect(parameterID, parameter.options);
            }
        }
        else if (parameter.type == "checkbox") {
            var extraAttributes = {
                class: "form-check-input",
                type: "checkbox",
                data_parameterName: parameterName
            };
            if (parameter.default) {
                extraAttributes.checked = true;
            }
            if (parameter.attributes !== undefined) {
                extraAttributes = {...extraAttributes, ...parameter.attributes};
            }
            var input11 = HTMLElementUtils.createElement({
                kind: "input",
                id: parameterID,
                extraAttributes: extraAttributes,
            });
            var label11 = HTMLElementUtils.createElement({
                kind: "label",
                extraAttributes: { for: parameterID },
            });
            label11.innerHTML = "&nbsp;" + parameter.label;
            col1.appendChild(input11);
            col1.appendChild(label11);
            row.appendChild(col1);
            activeDiv.appendChild(row);
        }
        if (parameter.type == "number" || parameter.type == "text" || parameter.type == "checkbox" || parameter.type == "select") {
            input11.addEventListener("change", (event) => {
                var parameterName = event.target.getAttribute("data_parameterName");
                var parameter = window[pluginID].parameters[parameterName];
                var value = undefined;
                if (parameter.type == "number") {
                    value = parseFloat(event.target.value);
                }
                else if (parameter.type == "text") {
                    value = event.target.value;
                }
                else if (parameter.type == "checkbox") {
                    value = event.target.checked;
                }
                else if (parameter.type == "select") {
                    value = event.target.value;
                }
                window[pluginID].set(parameterName, value);
                if (window[pluginID].inputTrigger !== undefined) {
                    window[pluginID].inputTrigger(parameterName);
                }
            });
        }
        window[pluginID].set(parameterName, parameter.default);
    }
    
}