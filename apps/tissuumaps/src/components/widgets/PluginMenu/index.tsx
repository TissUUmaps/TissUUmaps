import { EllipsisVerticalIcon } from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { pluginRegistry } from "@/plugins";
import { useAppStore } from "@/stores/app";

export type PluginMenuProps = {
  onShowPlugin?: (pluginId: string) => void;
  className?: string;
};

/**
 * The menu listing the registered plugins that have a user interface, by name
 *
 * Picking a plugin mounts it, which shows its panel; picking one that is
 * already mounted calls `onShowPlugin` instead, if given, to show its panel.
 */
export function PluginMenu({ onShowPlugin, className }: PluginMenuProps) {
  const plugins = useAppStore((state) => state.plugins);

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
                  onShowPlugin?.(pluginId);
                } else {
                  pluginRegistry.mountPlugin(pluginId);
                }
              }}
            >
              {plugin.name}
            </DropdownMenuItem>
          ))
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
