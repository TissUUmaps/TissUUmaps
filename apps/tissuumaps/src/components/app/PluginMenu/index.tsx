import { EllipsisVerticalIcon, FileIcon, LinkIcon } from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { getPluginPanelId } from "@/panels";
import { pluginRegistry } from "@/plugins";
import { useAppStore } from "@/stores/app";

import { useLoadPluginFromFile, useLoadPluginFromURL } from "./hooks";

export type PluginMenuProps = {
  className?: string;
};

/**
 * The menu listing the registered plugins that have a user interface, by name,
 * followed by the options for loading a third-party plugin
 *
 * Picking a plugin mounts it, which shows its panel; picking one that is
 * already mounted brings its panel to the front instead.
 */
export function PluginMenu({ className }: PluginMenuProps) {
  const plugins = useAppStore((state) => state.plugins);
  const setActivePanelId = useAppStore((state) => state.setActivePanelId);
  const loadPluginFromFile = useLoadPluginFromFile();
  const loadPluginFromURL = useLoadPluginFromURL();

  const mountablePlugins = [...plugins]
    .filter(([, plugin]) => plugin.mountable)
    .sort(([, a], [, b]) => a.name.localeCompare(b.name));

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <IconButton label="Plugins" size="icon" className={className} />
        }
      >
        <EllipsisVerticalIcon />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        {mountablePlugins.length === 0 ? (
          <DropdownMenuItem disabled>No plugins</DropdownMenuItem>
        ) : (
          mountablePlugins.map(([pluginId, plugin]) => (
            <DropdownMenuItem
              key={pluginId}
              onClick={() => {
                if (plugin.container !== undefined) {
                  setActivePanelId(getPluginPanelId(pluginId));
                } else {
                  pluginRegistry.mountPlugin(pluginId);
                }
              }}
            >
              {plugin.name}
            </DropdownMenuItem>
          ))
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={loadPluginFromFile}>
          <FileIcon />
          Load plugin from file…
        </DropdownMenuItem>
        <DropdownMenuItem onClick={loadPluginFromURL}>
          <LinkIcon />
          Load plugin from URL…
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
