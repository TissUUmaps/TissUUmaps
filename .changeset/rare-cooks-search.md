---
"@tissuumaps/core": minor
"tissuumaps": minor
---

Plugins are no longer mounted when registered: a menu in the tab bar lists the registered plugins and mounts the one picked, and closing a plugin's panel only unmounts it. The menu can also load third-party plugins, ES modules whose default export is the plugin, from a local file or a URL, absolute or relative to the page. Third-party plugins should both default-export the plugin and register it through `window.tissuumaps`, so that the same file can be loaded through the menu or included in the page. Adds `PluginRegistry.mountPlugin`/`unmountPlugin`.
