---
"@tissuumaps/render": minor
"tissuumaps": minor
---

A "Navigator" switch in the render settings of the Project panel shows or hides the navigator (the minimap in the viewer). It sets `osOptions.viewerOptions.showNavigator`, which is saved with the project. `OpenSeadragonContext` now applies `showNavigator` live, without rebuilding the viewer.
