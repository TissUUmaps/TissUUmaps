---
"@tissuumaps/core": minor
"tissuumaps": minor
---

The app store tracks the active panel and the expanded objects of each panel: `activePanelId` is the ID of the panel whose tab was selected last (the viewer, which has no tab, is never the active panel), and setting it with `setActivePanelId` brings that panel to the front; `expandedImageIds`, `expandedLabelsIds`, `expandedPointsIds`, `expandedShapesIds` and `expandedTableIds` are the IDs of the objects whose entries are expanded in the respective panels, set with `setExpandedImageIds` etc. and cleared whenever another project is loaded or the project is closed. These replace `showImageSettings`, `showLabelsSettings`, `showPointsSettings` and `showShapesSettings` and the `imageSettingsRequest`, `labelsSettingsRequest`, `pointsSettingsRequest` and `shapesSettingsRequest` state, which are removed: plugins that show an object's settings have to add its ID to the expanded IDs and set the active panel instead (e.g. `setExpandedImageIds([...expandedImageIds, imageId])` and `setActivePanelId("images")`).
