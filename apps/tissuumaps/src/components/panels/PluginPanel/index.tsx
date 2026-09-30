import { useEffect, useRef } from "react";

import { useAppStore } from "@/stores/app";

export type PluginPanelProps = {
  pluginId: string;
  className?: string;
};

/**
 * The panel of a mounted plugin
 *
 * Shows the element into which the plugin registry mounted the plugin's user
 * interface, as held by the app store. The panel only attaches that element;
 * mounting and unmounting are the registry's business, so that the user
 * interface is unmounted before the plugin is torn down, and so that it
 * survives the panel being hidden, moved or remounted.
 */
export function PluginPanel({ pluginId, className }: PluginPanelProps) {
  const ref = useRef<HTMLDivElement>(null);

  const container = useAppStore(
    (state) => state.plugins.get(pluginId)?.container,
  );

  useEffect(() => {
    const element = ref.current;
    if (element === null) {
      return;
    }
    if (container === undefined) {
      element.replaceChildren();
    } else if (container.parentElement !== element) {
      element.replaceChildren(container);
    }
  }, [container]);

  return <div ref={ref} className={className} />;
}
