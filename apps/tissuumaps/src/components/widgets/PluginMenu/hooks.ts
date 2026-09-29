import { useCallback } from "react";

import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { usePromptDialog } from "@/components/dialogs/PromptDialog/hooks";
import {
  loadPluginFromFile,
  loadPluginFromURL,
  pluginRegistry,
} from "@/plugins";

import { pickPluginFileFromInput } from "./pickPluginFileFromInput";

/** The warning shown before a third-party plugin is loaded */
const pluginTrustWarning =
  "A plugin runs with full access to TissUUmaps and the data it has opened. Only load plugins from sources you trust.";

/**
 * Returns a callback that lets the user pick a plugin module file and, after
 * confirmation, loads the plugin from it and opens it
 *
 * Failures are shown in an alert dialog and logged.
 *
 * @returns The callback
 */
export function useLoadPluginFromFile(): () => void {
  const confirm = useConfirmDialog();
  const alert = useAlertDialog();

  return useCallback(() => {
    void pickPluginFileFromInput().then(async (file) => {
      if (file === null) {
        return;
      }
      const confirmed = await confirm({
        title: "Load plugin",
        body: `Load the plugin from "${file.name}"? ${pluginTrustWarning}`,
        cancelLabel: "Cancel",
        actionLabel: "Load",
      });
      if (!confirmed) {
        return;
      }
      try {
        const pluginId = await loadPluginFromFile(file);
        if (pluginId !== null) {
          pluginRegistry.mountPlugin(pluginId);
        }
      } catch (error) {
        console.error(`Failed to load plugin from ${file.name}`, error);
        await alert({
          title: "Failed to load plugin",
          body: error instanceof Error ? error.message : String(error),
        });
      }
    });
  }, [alert, confirm]);
}

/**
 * Returns a callback that asks the user for a plugin module URL, and loads the
 * plugin from it and opens it
 *
 * The trust warning is shown in the prompt, so submitting the URL confirms
 * loading the plugin. Failures are shown in an alert dialog and logged.
 *
 * @returns The callback
 */
export function useLoadPluginFromURL(): () => void {
  const prompt = usePromptDialog();
  const alert = useAlertDialog();

  return useCallback(() => {
    void prompt({
      title: "Load plugin from URL",
      body: pluginTrustWarning,
      actionLabel: "Load",
      inputProps: { type: "url", required: true },
    }).then(async (value) => {
      const pluginUrl = value?.trim();
      if (!pluginUrl) {
        return;
      }
      try {
        const pluginId = await loadPluginFromURL(pluginUrl);
        if (pluginId !== null) {
          pluginRegistry.mountPlugin(pluginId);
        }
      } catch (error) {
        console.error(`Failed to load plugin from ${pluginUrl}`, error);
        await alert({
          title: "Failed to load plugin",
          body: error instanceof Error ? error.message : String(error),
        });
      }
    });
  }, [alert, prompt]);
}
