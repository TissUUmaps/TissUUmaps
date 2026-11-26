---
sidebar_position: 5
---

# Settings

Each point cloud, label image and shape cloud shows a **Settings** box in its entry in the Points, Labels or Shapes panel, below its data source. Without a table, the style rows are disabled. The other settings are under **Advanced**, collapsed at the end of the box.

- **Column** picks a column of the object's own table.
  - Tables with sections, such as AnnData, show one row per group, for example **Observations**, **Features** and **Embeddings**. Each row searches its columns by name. **from** chooses the set of columns of a row, for example `X`, a layer or `raw`; switching it keeps the feature if that set has it. A name found only in another set of the row is offered there, to switch to.
  - Other tables, such as CSV files, show one **Column** row.
  - Clearing the column returns every property that follows it to its single value: one color for points, random colors for labels. Settings made under **Advanced**, for example "Single color", show as such.
- The icon at the start of the picked column sets its **column type**: **Categorical** (tag) gives each value of the column its own color or marker. Its colors, and the sizes, opacities and visibility of its categories, are edited in the group table below the box. **Numerical** (#) uses the values themselves: colors through a palette, sizes and opacities as they are, and visibility hides items whose value is 0 or less. Categorical columns start as categorical, all others as numerical. Text columns, and columns the table declares categorical, can only be categorical. Using more than 1,000 distinct values as categorical asks first.
- **Applies to** chooses which properties follow the column: color, marker, size, opacity and visibility for points; shape fill, shape stroke, opacity and visibility for shapes; color, opacity and visibility for labels. The properties that follow the column keep following a newly picked one, as far as its type allows; if none can, the color (the fill color for shapes) starts. Markers need a categorical column; size, opacity and visibility a numerical one. One property always follows the column; clear the column to stop.
- For shapes, **Shape fill** and **Shape stroke** have their own toggles in their header for numerical columns: their color, opacity and visibility. For categorical columns, they follow by their color.
- A property that follows a column shows its options below. For a categorical column, colors have a **Palette**; it is disabled once a color is edited in the group table, which then sets every color. For a numerical column, colors have a **Palette** and a **Range** (an empty range end is the column's minimum or maximum), and sizes a **Unit**.
