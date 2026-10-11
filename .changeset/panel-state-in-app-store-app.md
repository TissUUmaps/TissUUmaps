---
"tissuumaps": minor
---

pr: #300
commit: 9fc85a9148be339aa0e419620253f03aace7041a

Breaking for plugins: the app store's `showImageSettings` etc. are removed. To show an object's settings, add its ID to `expandedImageIds` etc. and set `activePanelId`.
