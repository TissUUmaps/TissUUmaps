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
 }

pluginUtils.startPlugin = function (pluginID) {
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

        window[pluginID]["init"](pluginDiv);
    }
}
