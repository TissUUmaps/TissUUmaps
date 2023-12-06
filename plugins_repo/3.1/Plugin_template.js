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
};

/**
 * This method is called when the document is loaded.
 * The container element is a div where the plugin options will be displayed.
 * @summary After setting up the tmapp object, initialize it*/
Plugin_template.init = function (container) {
  row1 = HTMLElementUtils.createRow({});
  col11 = HTMLElementUtils.createColumn({ width: 12 });
  button111 = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-primary mx-2" },
  });
  button111.innerText = "Test plugin";

  button111.addEventListener("click", (event) => {
    interfaceUtils
      .prompt("Message", "Hello world", "Blabla")
      .then((pathFormat) => {
        Plugin_template.demo(pathFormat);
      });
  });

  container.innerHTML = "";
  container.appendChild(row1);
  row1.appendChild(col11);
  col11.appendChild(button111);
};

Plugin_template.demo = function (message) {
  console.log(
    JSON.stringify({
      message: message,
    }),
  );
  $.ajax({
    type: "post",
    url: "/plugins/Plugin_template/server_demo",
    contentType: "application/json; charset=utf-8",
    data: JSON.stringify({
      message: message,
    }),
    success: function (data) {
      interfaceUtils.alert(data);
    },
    complete: function (data) {
      // do something, not critical.
    },
    error: function (data) {
      console.log("Error:", data);
    },
  });
};
