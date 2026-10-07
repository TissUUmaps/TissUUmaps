import { type DataSource, SourceUtils } from "@tissuumaps/core";

import type { AddDataObjectDialogParams } from "@/components/dialogs/AddDataObjectDialog";
import {
  useAddDataObjectDialog,
  useAddImageDialogParams,
  useAddLabelsDialogParams,
  useAddPointsDialogParams,
  useAddShapesDialogParams,
  useAddTableDialogParams,
} from "@/components/dialogs/AddDataObjectDialog/hooks";
import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { useLoadProjectFile } from "@/components/panels/ProjectPanel/hooks";
import { readDroppedItems, resolveWorkspacePath } from "@/data/io/workspace";
import { PanelId } from "@/panels";
import { useAppStore } from "@/stores/app";
import { projectStore, useProjectStore } from "@/stores/project";

/**
 * Returns what a panel accepts when files or directories are dropped on it,
 * and handles the drop
 *
 * The project panel accepts a single project file, which it loads, or a
 * single directory, which it opens as the workspace, either after
 * confirmation. The data panels accept files and directories within the
 * workspace, and open one add data object dialog per item; they accept
 * nothing while no project or workspace is open, while no data provider of
 * their kind is registered, or while their data objects need a layer and
 * there is none.
 *
 * @param panelId - The dockview ID of the panel
 * @returns The label of the drop target, the names of what it accepts, and
 * the drop handler, which has to be called within the drop event; `null` if
 * the panel accepts nothing
 */
export function usePanelDrop(panelId: string): {
  label: string;
  accepts: string[];
  onDrop: (dataTransfer: DataTransfer) => Promise<void>;
} | null {
  const workspace = useAppStore((state) => state.workspace);
  const isProjectOpen = useAppStore((state) => state.projectOpen);
  const setWorkspace = useAppStore((state) => state.setWorkspace);
  const loadProjectFile = useLoadProjectFile();
  const hasLayers = useProjectStore((state) => state.layers.length > 0);
  const alert = useAlertDialog();
  const confirm = useConfirmDialog();
  const addDataObject = useAddDataObjectDialog();
  const addImageDialogParams = useAddImageDialogParams();
  const addLabelsDialogParams = useAddLabelsDialogParams();
  const addPointsDialogParams = useAddPointsDialogParams();
  const addShapesDialogParams = useAddShapesDialogParams();
  const addTableDialogParams = useAddTableDialogParams();

  const dropProject = async (dataTransfer: DataTransfer) => {
    const items = await readDroppedItems(dataTransfer);
    const item = items[0];
    if (item === undefined || items.length > 1) {
      await alert({
        title: "Cannot open the dropped items",
        body: "Drop a single project file or folder.",
      });
      return;
    }
    if (item.handle?.kind === "directory") {
      const directory = item.handle as FileSystemDirectoryHandle;
      if (workspace !== null && (await workspace.isSameEntry(directory))) {
        await alert({
          title: "Folder already open",
          body: `The folder "${directory.name}" is already open.`,
        });
        return;
      }
      const confirmed = await confirm({
        title: "Open folder",
        body:
          workspace !== null
            ? `Open the folder "${directory.name}" instead of "${workspace.name}"?`
            : `Open the folder "${directory.name}"?`,
        actionLabel: "Open",
      });
      if (confirmed) {
        setWorkspace(directory);
      }
      return;
    }
    const name = item.handle?.name ?? item.file?.name ?? "";
    const confirmed = await confirm({
      title: "Open project",
      body: isProjectOpen
        ? `Open the project "${name}"? It replaces the open project, and unsaved changes are lost.`
        : `Open the project "${name}"?`,
      actionLabel: "Open",
    });
    if (!confirmed) {
      return;
    }
    try {
      const projectFile =
        item.handle?.kind === "file"
          ? (item.handle as FileSystemFileHandle)
          : item.file;
      if (projectFile === null) {
        throw new Error("The dropped item cannot be read.");
      }
      await loadProjectFile(projectFile);
    } catch (error) {
      console.error("Failed to load the dropped project", error);
      await alert({
        title: "Cannot open the project",
        body: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const getDataDrop = <TDataSource extends DataSource>(
    params: AddDataObjectDialogParams<TDataSource>,
  ) => {
    if (
      !isProjectOpen ||
      workspace === null ||
      params.dataProviders.size === 0 ||
      (params.withLayer === true && !hasLayers)
    ) {
      return null;
    }
    return {
      label: `Drop to ${params.title.toLowerCase()}`,
      accepts: Array.from(
        params.dataProviders.values(),
        (dataProvider) => dataProvider.name,
      ),
      onDrop: async (dataTransfer: DataTransfer) => {
        const items = await readDroppedItems(dataTransfer);
        const projectSource = projectStore.getState().source;
        const itemSources = await Promise.all(
          items.map(async (item) => {
            if (item.handle === null) {
              return null;
            }
            try {
              const workspacePath = await resolveWorkspacePath(
                item.handle,
                workspace,
              );
              return SourceUtils.makeProjectPath(workspacePath, projectSource);
            } catch (error) {
              console.warn(`Skipped the dropped "${item.handle.name}"`, error);
              return null;
            }
          }),
        );
        const sources = itemSources.filter((source) => source !== null);
        const skippedNames = items
          .filter((_, i) => itemSources[i] === null)
          .map((item) => item.handle?.name ?? item.file?.name ?? "");
        if (skippedNames.length > 0) {
          await alert({
            title: "Cannot add some of the dropped items",
            body: `Only files and folders within the connected folder can be added. These dropped items are skipped: ${skippedNames.join(", ")}`,
          });
        }
        addDataObject(params, sources);
      },
    };
  };

  switch (panelId) {
    case PanelId.project:
      return {
        label: "Drop to open",
        accepts: ["Project file", "Folder (workspace)"],
        onDrop: dropProject,
      };
    case PanelId.images:
      return getDataDrop(addImageDialogParams);
    case PanelId.labels:
      return getDataDrop(addLabelsDialogParams);
    case PanelId.points:
      return getDataDrop(addPointsDialogParams);
    case PanelId.shapes:
      return getDataDrop(addShapesDialogParams);
    case PanelId.tables:
      return getDataDrop(addTableDialogParams);
    default:
      return null;
  }
}
