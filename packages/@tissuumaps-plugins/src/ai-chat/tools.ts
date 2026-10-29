import { callAction } from "./call-action";
import {
  colorByColumn,
  setColor,
  setGroupColors,
  showAllGroups,
  showOnlyGroups,
} from "./color-tools";
import {
  setOpacity,
  setPointSize,
  setVisibility,
  showOnly,
} from "./object-tools";
import { itemsKinds, objectKinds } from "./objects";
import type { Tool } from "./types";

/** The ids parameter that most tools share */
const idsParameter = {
  type: "array",
  items: { type: "string" },
  description:
    "The ids of the objects, exactly as in the project summary. Put several objects in one list.",
};

/** The category parameter, for all the objects of a section of the summary */
const categoryParameter = {
  type: "string",
  enum: objectKinds,
  description:
    'Instead of ids: all the objects of this section of the summary, e.g. "points" for all points.',
};

/** The type parameter, for ids that are in more than one section */
const typeParameter = {
  type: "string",
  enum: objectKinds,
  description:
    "Optional. Only needed when the same id is in more than one section of the summary.",
};

/** The category parameter of the color and filter tools */
const itemsCategoryParameter = {
  type: "string",
  enum: itemsKinds,
  description: "Instead of ids: all the labels or all the points.",
};

/** The column parameter of the tools that work on the groups of a column */
const groupsColumnParameter = {
  type: "string",
  description: "Optional. Without it: the column the object is colored by now.",
};

/** The color names the model may use */
const colorNamesDescription =
  "red, green, blue, yellow, orange, purple, pink, brown, cyan, magenta, white, black, grey, light grey, dark grey";

/** The tools, as described to the model */
export const toolDefinitions = [
  {
    type: "function",
    function: {
      name: "set_visibility",
      description:
        "Show or hide objects: the ones in ids, or all the objects of a category.",
      parameters: {
        type: "object",
        required: ["visible"],
        properties: {
          ids: idsParameter,
          category: categoryParameter,
          visible: {
            type: "boolean",
            description: "true to show, false to hide",
          },
          type: typeParameter,
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "show_only",
      description:
        "Show only the given objects (ids or a whole category) and hide all the other images, labels, points and shapes. Without ids and category it hides everything.",
      parameters: {
        type: "object",
        required: [],
        properties: {
          ids: idsParameter,
          category: categoryParameter,
          type: typeParameter,
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "set_opacity",
      description:
        "Set the opacity of objects (ids or a whole category), from 0 (invisible) to 1 (fully opaque).",
      parameters: {
        type: "object",
        required: ["opacity"],
        properties: {
          ids: idsParameter,
          category: categoryParameter,
          opacity: {
            type: "number",
            description:
              'From 0 to 1. "50% transparent" is 0.5, "30% transparent" is 0.7, "80% opaque" is 0.8.',
          },
          type: typeParameter,
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "set_point_size",
      description:
        'Make the points of points objects (ids, or category "points" for all) bigger or smaller than they are now, by a percentage.',
      parameters: {
        type: "object",
        required: ["change_percent"],
        properties: {
          ids: idsParameter,
          category: {
            type: "string",
            enum: ["points"],
            description: 'Use "points" for all the points objects.',
          },
          change_percent: {
            type: "number",
            description:
              'The percentage the user said, positive for bigger, negative for smaller. Do no math: "10% smaller" is -10, "50% bigger" is 50, "twice as big" is 100, "half the size" is -50.',
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "color_by_column",
      description:
        "Color labels or points by a column of their own table: each group gets its own color, numbers get a color scale.",
      parameters: {
        type: "object",
        required: ["column"],
        properties: {
          ids: idsParameter,
          category: itemsCategoryParameter,
          column: {
            type: "string",
            description:
              'A column from "columns you can use" of the object, e.g. "leiden", or a gene name, e.g. "EPCAM".',
          },
          palette: {
            type: "string",
            description:
              'Only if the user names one, e.g. "viridis" or "tableau10".',
          },
          range: {
            type: "array",
            items: { type: "number" },
            description:
              "Only for numbers, and only if the user gives one: [lowest, highest], e.g. [0, 500].",
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "set_group_colors",
      description:
        'Give chosen colors to some groups of labels or points, e.g. "cluster 3 in red, the others grey".',
      parameters: {
        type: "object",
        required: ["colors"],
        properties: {
          ids: idsParameter,
          category: itemsCategoryParameter,
          colors: {
            type: "object",
            additionalProperties: { type: "string" },
            description: `Group → color, e.g. {"3": "red", "T cell": "blue"}. Colors in English (${colorNamesDescription}) or "#rrggbb".`,
          },
          others: {
            type: "string",
            description:
              "The color of all the other groups. Leave it out if the user does not say it: they become light grey.",
          },
          column: groupsColumnParameter,
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "set_color",
      description:
        'Give one color to all the items of labels or points, e.g. "color the cell circles green".',
      parameters: {
        type: "object",
        required: ["color"],
        properties: {
          ids: idsParameter,
          category: itemsCategoryParameter,
          color: {
            type: "string",
            description: `A color in English (${colorNamesDescription}) or "#rrggbb".`,
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "show_only_groups",
      description:
        'Show only some groups of labels or points, e.g. "only clusters 1 and 3", and hide their other items.',
      parameters: {
        type: "object",
        required: ["groups"],
        properties: {
          ids: idsParameter,
          category: itemsCategoryParameter,
          groups: {
            type: "array",
            items: { type: "string" },
            description:
              'The groups to show, e.g. ["1", "3"] or ["EPCAM", "KRT7"].',
          },
          column: groupsColumnParameter,
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "show_all_groups",
      description:
        "Remove the group filter of labels or points: all their items are shown again.",
      parameters: {
        type: "object",
        required: [],
        properties: {
          ids: idsParameter,
          category: itemsCategoryParameter,
        },
      },
    },
  },
];

/** The function that runs each tool, by tool name */
export const tools: Record<string, Tool> = {
  set_visibility: setVisibility,
  show_only: showOnly,
  set_opacity: setOpacity,
  set_point_size: setPointSize,
  color_by_column: colorByColumn,
  set_group_colors: setGroupColors,
  set_color: setColor,
  show_only_groups: showOnlyGroups,
  show_all_groups: showAllGroups,
  // only offered to OpenAI and Anthropic, see runTool
  call_action: callAction,
};
