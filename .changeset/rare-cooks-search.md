---
"@tissuumaps/core": minor
"tissuumaps": minor
---

Plugins are no longer mounted when registered: a menu in the tab bar lists the registered plugins and mounts the one picked, and closing a plugin's panel only unmounts it. Adds `PluginRegistry.mountPlugin`/`unmountPlugin`.
