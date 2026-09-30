---
"@tissuumaps/core": minor
"tissuumaps": minor
---

Plugins can show the settings of a project object: `appStore.getState().focusObject({ kind, id })` brings the panel of the object's collection to the front and expands the object's settings. Adds `AppStoreActions.focusObject`, `AppStoreState.focusedObject` and the `FocusedObject` type.
