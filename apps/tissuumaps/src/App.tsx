import {
  type DockviewApi,
  DockviewDefaultTab,
  DockviewReact,
  type DockviewReadyEvent,
  type DockviewTheme,
  type IDockviewHeaderActionsProps,
  type IDockviewPanelHeaderProps,
  type IDockviewPanelProps,
} from "dockview-react";
import { Moon, Sun } from "lucide-react";
import { type ReactNode, useLayoutEffect, useState } from "react";

import { IconButton } from "@/components/common/icon-button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useFocusedObjectPanel } from "@/hooks/useFocusedObject";
import { getPluginPanelId, usePluginPanels } from "@/hooks/usePluginPanels";

import "./App.css";
import { DialogProvider } from "./components/dialogs/DialogProvider";
import { ImagesPanel } from "./components/panels/ImagesPanel";
import { LabelsPanel } from "./components/panels/LabelsPanel";
import { PluginPanel } from "./components/panels/PluginPanel";
import { PointsPanel } from "./components/panels/PointsPanel";
import { ProjectPanel } from "./components/panels/ProjectPanel";
import { ShapesPanel } from "./components/panels/ShapesPanel";
import { TablesPanel } from "./components/panels/TablesPanel";
import { ViewerPanel } from "./components/panels/ViewerPanel";
import { PanelId } from "./components/panels/panelId";
import { NotificationCenter } from "./components/widgets/NotificationCenter";
import { PluginMenu } from "./components/widgets/PluginMenu";
import { pluginRegistry } from "./plugins";
import { useSettingsStore } from "./stores/settings";

/** The Tailwind CSS-styled dockview theme defined in `dockview.css` */
const dockviewTheme: DockviewTheme = {
  name: "tailwindcss",
  className: "dockview-theme-tailwindcss",
};

/**
 * Scrolls a panel's content within the panel, rather than letting it overflow
 *
 * The spacing around the content is padding rather than the content's own
 * margin, which a scroll container would cut off at its bottom end. Scrolling
 * one axis makes the browser scroll the other one too unless it is hidden, and
 * the widgets that are too wide for a panel scroll horizontally by themselves.
 */
function ScrollablePanelContent({ children }: { children: ReactNode }) {
  return (
    <div className="size-full overflow-x-hidden overflow-y-auto p-2">
      {children}
    </div>
  );
}

/** The panels that can be shown in the dockview layout, by component name */
const dockviewComponents = {
  ViewerPanel: () => <ViewerPanel className="size-full" />,
  ProjectPanel: (props: IDockviewPanelProps) => (
    <ScrollablePanelContent>
      <ProjectPanel
        onShowPanel={(panelId) =>
          props.containerApi.getPanel(panelId)?.api.setActive()
        }
      />
    </ScrollablePanelContent>
  ),
  ImagesPanel: () => (
    <ScrollablePanelContent>
      <ImagesPanel />
    </ScrollablePanelContent>
  ),
  LabelsPanel: () => (
    <ScrollablePanelContent>
      <LabelsPanel />
    </ScrollablePanelContent>
  ),
  PointsPanel: () => (
    <ScrollablePanelContent>
      <PointsPanel />
    </ScrollablePanelContent>
  ),
  ShapesPanel: () => (
    <ScrollablePanelContent>
      <ShapesPanel />
    </ScrollablePanelContent>
  ),
  TablesPanel: () => (
    <ScrollablePanelContent>
      <TablesPanel />
    </ScrollablePanelContent>
  ),
  PluginPanel: (props: IDockviewPanelProps<{ pluginId: string }>) => (
    <ScrollablePanelContent>
      <PluginPanel pluginId={props.params.pluginId} />
    </ScrollablePanelContent>
  ),
};

/**
 * The tab headers available to panels, by component name: one that lets the
 * user close the panel, one for panels that are always shown, and one for the
 * panels contributed by plugins, whose close button unmounts the plugin
 */
const dockviewTabComponents = {
  ClosablePanelHeader: (props: IDockviewPanelHeaderProps) => {
    return <DockviewDefaultTab hideClose={false} {...props} />;
  },
  PersistentPanelHeader: (props: IDockviewPanelHeaderProps) => {
    return <DockviewDefaultTab hideClose={true} {...props} />;
  },
  PluginPanelHeader: (
    props: IDockviewPanelHeaderProps<{ pluginId: string }>,
  ) => {
    return (
      <DockviewDefaultTab
        {...props}
        hideClose={false}
        closeActionOverride={() =>
          pluginRegistry.unmountPlugin(props.params.pluginId)
        }
      />
    );
  },
};

/**
 * The plugins menu and the dark mode toggle shown at the right end of the
 * dockview tab bar
 */
function DockviewRightHeaderActionsComponent(
  props: IDockviewHeaderActionsProps,
) {
  const dark = useSettingsStore((state) => state.dark);
  const setDark = useSettingsStore((state) => state.setDark);
  return (
    <div className="flex">
      <PluginMenu
        onShowPlugin={(pluginId) =>
          props.containerApi
            .getPanel(getPluginPanelId(pluginId))
            ?.api.setActive()
        }
      />
      <IconButton
        label={dark ? "Light mode" : "Dark mode"}
        variant="default"
        size="icon"
        onClick={() => setDark(!dark)}
      >
        {dark ? <Sun /> : <Moon />}
      </IconButton>
    </div>
  );
}

/**
 * Creates the application's initial panel layout
 *
 * The viewer panel fills the window; its group is locked and its header hidden,
 * so that it cannot be closed or moved. All other panels share a group to its
 * right, with the project panel active.
 *
 * @param event - The event carrying the API of the ready dockview
 */
const onDockviewReady = (event: DockviewReadyEvent) => {
  const viewerPanel = event.api.addPanel({
    id: PanelId.viewer,
    title: "Viewer",
    component: "ViewerPanel",
  });
  viewerPanel.group.header.hidden = true;
  viewerPanel.group.locked = true;
  const projectPanel = event.api.addPanel({
    id: PanelId.project,
    title: "Project",
    component: "ProjectPanel",
    tabComponent: "PersistentPanelHeader",
    initialWidth: 420,
    position: {
      referencePanel: viewerPanel,
      direction: "right",
    },
  });
  event.api.addPanel({
    id: PanelId.images,
    title: "Images",
    component: "ImagesPanel",
    tabComponent: "PersistentPanelHeader",
    position: { referenceGroup: projectPanel.group },
  });
  event.api.addPanel({
    id: PanelId.labels,
    title: "Labels",
    component: "LabelsPanel",
    tabComponent: "PersistentPanelHeader",
    position: { referenceGroup: projectPanel.group },
  });
  event.api.addPanel({
    id: PanelId.points,
    title: "Points",
    component: "PointsPanel",
    tabComponent: "PersistentPanelHeader",
    position: { referenceGroup: projectPanel.group },
  });
  event.api.addPanel({
    id: PanelId.shapes,
    title: "Shapes",
    component: "ShapesPanel",
    tabComponent: "PersistentPanelHeader",
    position: { referenceGroup: projectPanel.group },
  });
  event.api.addPanel({
    id: PanelId.tables,
    title: "Tables",
    component: "TablesPanel",
    tabComponent: "PersistentPanelHeader",
    position: { referenceGroup: projectPanel.group },
  });
  projectPanel.api.setActive();
};

/**
 * The application's root component
 *
 * Renders the dockview layout within the app-level providers, and applies
 * Tailwind CSS's `dark` class to the document element according to the
 * settings store, so that inherited text colors and the popups portalled into
 * the body follow it too.
 */
export function App() {
  const dark = useSettingsStore((state) => state.dark);
  const [dockviewApi, setDockviewApi] = useState<DockviewApi | null>(null);

  // The panels contributed by plugins join the group of the project panel
  usePluginPanels(dockviewApi, PanelId.project);
  useFocusedObjectPanel(dockviewApi);

  // Before paint, so that React never renders a light frame in dark mode
  // https://tailwindcss.com/docs/dark-mode
  useLayoutEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  return (
    <TooltipProvider delay={300}>
      <DialogProvider>
        <div className="w-screen h-screen overflow-hidden">
          <DockviewReact
            theme={dockviewTheme}
            components={dockviewComponents}
            tabComponents={dockviewTabComponents}
            rightHeaderActionsComponent={DockviewRightHeaderActionsComponent}
            onReady={(event) => {
              onDockviewReady(event);
              setDockviewApi(event.api);
            }}
          />
          <NotificationCenter />
        </div>
      </DialogProvider>
    </TooltipProvider>
  );
}
