/**
 * @file Creation of custom elements
 * @see {@link customElementUtils}
 */
/**
 * @namespace customElementUtils
 */

customElementUtils = {};

/**
 * @param id Input id
 * @param listId Option list id
 * @param extraAttributes Extra HTML attributes to be applied to the input
 * @param options Options to be included in the drop down list
 * @param eventListeners Key value pairs of event names and event handler functions
 * @summary Create a select type element with text input
 */
customElementUtils.textInputSelect = function (
  id,
  listId,
  extraAttributes,
  options,
  eventListeners
) {
  // Create HTML elements
  const div = document.createElement("div");
  const list = document.createElement("datalist");
  const input = document.createElement("input");
  // Set ids if provided, they are needed if option population is
  // to be done after the component´s creation
  if (id && listId) {
    input.setAttribute("id", id);
    list.setAttribute("id", listId);
    input.setAttribute("name", id);
  }
  // Set styles
  div.style.position = "relative";
  input.style.width = "100%";
  // Set extra attributes to input element
  if (extraAttributes) {
    for (let attribute in extraAttributes) {
      input.setAttribute(attribute, extraAttributes[attribute]);
    }
  }
  input.setAttribute(
    "class",
    input.getAttribute("class") + " form-select form-select-sm"
  );
  // Populate list options
  if (options) {
    options.forEach(function (symbol, i) {
      const option = document.createElement("option");
      if (symbol.text) {
        option.value = symbol.value;
        option.text = symbol.text;
      } else {
        option.value = i;
        option.text = symbol;
      }
      list.appendChild(option);
    });
  }
  // Add event listeners
  if (eventListeners) {
    for (let event in eventListeners) {
      input.addEventListener(event, eventListeners[event]);
    }
  }

  div.appendChild(input);
  div.appendChild(list);

  // Detect changes in the option list and add click handlers to the options
  observer = new MutationObserver(function (mutationsList, observer) {
    // Set input value to "null" by default to maintain interoperability with
    // dataUtils.js dropdown data retrieval logic and checks
    if (input.value === "") {
      input.value = "null";
    }
    for (child in list.children) {
      list.children[child].onclick = function () {
        input.value = this.value;
        list.style.display = "none";
        unsetActiveOption(list.children);
        this.classList.add("active-option");
      };
    }
  });
  observer.observe(list, {
    characterData: false,
    childList: true,
    attributes: false,
  });

  let currentFocus = -1;

  // Set event handlers

  // If user clicks outside, close dropdown
  document.body.addEventListener("click", (event) => {
    if (!div.contains(event.target)) {
      list.style.display = "none";
      for (child in list.children) {
        if (!list.children[child].style) continue;
        list.children[child].style.display = "block";
      }
    }
  });

  // Open dropdown on focus
  input.onfocus = () => {
    if (input.value === "null") input.value = "";
    list.style.display = "block";
    list.style.width = input.clientWidth + "px";
    for (child in list.children) {
      if (!list.children[child].classList) continue;
      if (
        !Array.from(list.children[child].classList)?.includes("active-option")
      )
        continue;
      scrollOptionToTop(list.children[child], child);
    }
  };
  // Remove text if it does not match any option
  // when the input loses focus
  input.onblur = (event) => {
    if (!event.target.value) {
      input.value = "null";
      return;
    }
    const listOptions = Array.from(list.childNodes).map(
      (option) => option.innerText
    );
    if (!listOptions.includes(event.target.value)) {
      input.value = "null";
      unsetActiveOption(list.children);
    }
  };
  // Open dropdown on click
  input.onclick = () => {
    if (list.style.display === "block") return;
    list.style.display = "block";
    list.style.width = input.clientWidth + "px";
  };
  // Filter and set active option as the user types
  input.oninput = () => {
    if (list.style.display !== "block") {
      list.style.display = "block";
      list.style.width = input.clientWidth + "px";
    }
    const text = input.value.toLowerCase().trim();
    for (child in list.children) {
      if (list.children[child].value?.toLowerCase().trim().includes(text)) {
        list.children[child].style.display = "block";
      } else {
        if (list.children[child].style) {
          list.children[child].style.display = "none";
        }
      }
    }
    currentFocus = 0;
    setActiveOption(getVisibleOptions());
  };
  // Handle arrow and enter keys behaviour
  input.onkeydown = (event) => {
    if (event.code == "ArrowDown") {
      if (currentFocus === getVisibleOptions().length - 1) return;
      currentFocus++;
      setActiveOption(getVisibleOptions());
      arrowKeyScroll(getVisibleOptions()[currentFocus], currentFocus);
    }
    if (event.code == "ArrowUp") {
      if (currentFocus === 0) return;
      currentFocus--;
      setActiveOption(getVisibleOptions());
      arrowKeyScroll(getVisibleOptions()[currentFocus], currentFocus);
      return;
    }
    if (event.code == "Enter") {
      event.preventDefault();
      if (currentFocus > -1) {
        /*and simulate a click on the "active" item:*/
        if (list.children) getVisibleOptions()[currentFocus].click();
      }
    }
  };

  // Scrolls options for arrow key navigation
  function arrowKeyScroll(option, optionIndex) {
    const optionHeight = option.clientHeight;
    const scrollTo = optionIndex * optionHeight;
    // Only ten options are displayed due to
    // the list and options height, so check if it
    // is equal or surpases the height of ten options
    if (scrollTo > list.scrollTop + optionHeight * 9) {
      list.scrollTop = list.scrollTop + optionHeight;
    }
    if (scrollTo < list.scrollTop) {
      list.scrollTop = scrollTo;
    }
  }
  // Scrolls an option to the top of the list
  function scrollOptionToTop(option, optionIndex) {
    const optionHeight = option.clientHeight;
    const scrollTo = optionIndex * optionHeight;
    list.scrollTop = scrollTo;
  }
  // Returns visible options at any given time
  function getVisibleOptions() {
    return Array.from(list.children).filter(
      (child) => child.style.display !== "none"
    );
  }
  // Set/Unset active option
  function setActiveOption(options) {
    if (!options[currentFocus]) return;
    unsetActiveOption(list.children);
    if (currentFocus >= options.length) currentFocus = 0;
    if (currentFocus < 0) currentFocus = options.length - 1;
    options[currentFocus].classList.add("active-option");
  }
  function unsetActiveOption(options) {
    for (var i = 0; i < options.length; i++) {
      options[i].classList.remove("active-option");
    }
  }
  return div;
};

customElementUtils.dropDownButton = function(params, text, options) {
    if (!params) {
        console.log(params)
        const button = document.createElement("button");
        button.innerHTML = "unnamed";
        return button;
    }
    const container = document.createElement("div");
    const list = document.createElement("ul");
    const button = document.createElement("button");
    button.innerText = text;
    button.setAttribute("aria-expanded", false);
    button.setAttribute("data-bs-toggle", "dropdown");
    (params.id || null ? button.setAttribute("id", params.id) : null);
    (params.innerText || null ? button.innerHTML = params.innerText : null);
    (params["class"] || null ? button.setAttribute("class", params["class"]) : null);
   
    const eventListeners = params.eventListeners
    if (eventListeners) {
        for (let message in eventListeners) {
            button.addEventListener(message, eventListeners[message]);
        }
    }
    if (params.extraAttributes) {
        for (let attr in params.extraAttributes) {
            button.setAttribute(attr, params.extraAttributes[attr]);
        }
    }
    list.classList.add("dropdown-menu")
    button.classList.add("dropdown-toggle");
    button.style.height = "100%";
    if(options){
        options.forEach((option) => {
            const optionItem = document.createElement("li");
            optionItem.innerHTML = option.text;
            optionItem.id = option.id;
            optionItem.classList.add("dropdown-item");
            optionItem[option.event] = option.handler;
            list.appendChild(optionItem)
        })
    }
    container.appendChild(button)
    container.appendChild(list)
    return container;
}