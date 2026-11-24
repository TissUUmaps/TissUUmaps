---
sidebar_position: 5
---

# Style section

Each point cloud, label image and shape cloud with a table shows a **style section** at the top of its entry in the Points, Labels or Shapes panel. It holds the settings needed most often; everything else is under **Advanced settings**, collapsed below it.

- **Color by**: the column of the object's own table to color by.
  - Tables that group their columns, such as AnnData, show one row per group, named by the table (groups may also nest, shown as sections of **from**): **Annotations** for the columns describing the observations (e.g. `obs`), and **Features** for the columns of a matrix (e.g. genes in `X`), searched by name. When a group holds several sets of columns, **from** chooses one (e.g. `X`, a layer or `raw`); switching it keeps the colored feature if the table holds it there too, and otherwise says so.
  - Other tables, such as CSV files, show a single **Column** row.
  - Clearing the column colors by none: a uniform color for points and shapes, and random colors for labels. Other colorings without a column, set in the advanced settings, are shown as such (e.g. "Single color").
- **Show as**: **Categories** gives each distinct value a color, listed in the group table below the section, where colors (and markers) can be edited; **Gradient** maps numeric values to a palette. Numeric columns start as a gradient, all others as categories; only numeric columns can be shown as a gradient. A column with more than 1,000 distinct values is shown as categories only after confirmation. Picking another column keeps the palette.
- For a gradient, the minimum and maximum default to the range of the data; editing either fixes the range, clearing one returns it to the data, and the reset button returns to the data range.
