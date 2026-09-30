---
"@tissuumaps/core": minor
"tissuumaps": minor
---

Plugins can show the settings of an image, labels, points or shapes: `appStore.getState().showImageSettings(imageId)` brings the Images panel to the front and expands the image's settings, and `showLabelsSettings`, `showPointsSettings` and `showShapesSettings` do the same for the other panels. Adds these actions and the `imageSettingsRequest`, `labelsSettingsRequest`, `pointsSettingsRequest` and `shapesSettingsRequest` app store state.
