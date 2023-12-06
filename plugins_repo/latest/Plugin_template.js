/**
 * @file Plugin_template.js
 * @author Christophe Avenel
 */

/**
 * @namespace Plugin_template
 * @classdesc The root namespace for Plugin_template.
 */
var Plugin_template;
Plugin_template = {
  name: "Template Plugin",
  parameters: {
    _section_test: {
      label: "Test section",
      title: "Section 1",
      type: "section",
      collapsed: false,
    },
    _message: {
      label: "Message",
      type: "text",
      default: "Hello world",
    },
    _testButton: {
      label: "Test button",
      type: "button",
    },
  },
};

/**
 * This method is called when the document is loaded.
 * The container element is a div where the plugin options will be displayed.
 * @summary After setting up the tmapp object, initialize it*/
Plugin_template.init = function (container) {
  interfaceUtils.alert("The plugin has been loaded");
};

/**
 * This method is called when a button is clicked or a parameter value is changed*/
Plugin_template.inputTrigger = function (input) {
  console.log("inputTrigger", input);
  if (input === "_testButton") {
    let message = Plugin_template.get("_message");
    Plugin_template.demo(message);
  }
};

Plugin_template.demo = function (message) {
  let successCallback = function (data) {
    interfaceUtils.alert(data);
  };
  let errorCallback = function (data) {
    console.log("Error:", data);
  };
  // Call the Python API endpoint "server_demo"
  Plugin_template.api(
    "server_demo",
    { message: message },
    successCallback,
    errorCallback,
  );
};
