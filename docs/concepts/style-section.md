---
sidebar_position: 5
---

# Style section

Each point cloud, label image and shape cloud shows a **style section** at the top of its entry in the Points, Labels or Shapes panel. Without a table, it is disabled. The other settings are under **Advanced settings**, collapsed below it.

- **Color by** picks a column of the object's own table.
  - Tables with query groups, such as AnnData, show one row per group, for example **Annotations**, **Features** and **Embeddings**. Each row searches its columns by name. **from** chooses the set of columns of a row, for example `X`, a layer or `raw`; switching it keeps the colored feature if that set has it. A name found only in another set of the row is offered there, to switch to.
  - Other tables, such as CSV files, show one **Column** row.
  - Clearing the column colors points with one color and labels with random colors, and hides the fill of shapes. Other colorings set in the advanced settings show as such, for example "Single color".
- **Show as**: **Categories** gives each value a color, edited in the group table below the section. **Gradient** maps numeric values to a palette. Categorical columns start as categories, all others as a gradient. Categorical columns cannot be shown as a gradient. Showing more than 1,000 distinct values as categories asks first. Picking another column keeps the palette.
- **Markers by category** (points, categories only): gives each category its own marker too. Once switched on, the markers follow the colored column.
- **Palette** and **Range** set the gradient. An empty range end is the column's minimum or maximum.
