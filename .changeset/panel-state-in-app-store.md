---
"@tissuumaps/core": minor
---

The app store tracks the active panel (`activePanelId`) and the expanded objects of each panel (`expandedImageIds` etc.).

Breaking: `showImageSettings` etc. and the `imageSettingsRequest` etc. state are removed. Plugins call `setExpandedImageIds` etc. and `setActivePanelId` instead.
