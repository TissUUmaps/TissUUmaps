import {
  type DataSource,
  type ProjectProvider,
  SourceUtils,
} from "@tissuumaps/core";

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
import { useAskDialog } from "@/components/dialogs/AskDialog/hooks";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import {
  useChooseProjectProvider,
  useLoadProjectFile,
} from "@/components/panels/ProjectPanel/hooks";
import {
  findLocalProjectProviders,
  getProjectFileExtensions,
  importProjectFromFile,
  importProjectFromFileHandle,
  resolveProjectSource,
} from "@/data/io/project";
import {
  isWorkspaceSupported,
  readDroppedItems,
  resolveWorkspacePath,
} from "@/data/io/workspace";
import { PanelId } from "@/panels";
import { useAppStore } from "@/stores/app";
import { projectStore, useProjectStore } from "@/stores/project";

/**
 * Returns what a panel accepts when files or directories are dropped on it,
 * and handles the drop
 *
 * The project panel accepts a single project file, which it opens, or
 * imports into the open project if the user chooses so, or a single
 * directory, which it opens as the workspace, either after confirmation.
 * Without an open project, a file that no project provider or several load
 * by its extension is confirmed by choosing the project provider (see
 * `useChooseProjectProvider`). It warns about project files outside the
 * workspace when asking.
 * The data panels accept files and directories within the workspace, and
 * open one add data object dialog per item, also while no project is open,
 * which then opens once a data object is added. They accept nothing while no
 * workspace is open, while no data provider of their kind is registered, or
 * while their data objects need a layer and there is none.
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

  const hasLayers = useProjectStore((state) => state.layers.length > 0);

  const loadProjectFile = useLoadProjectFile();

  const chooseProjectProvider = useChooseProjectProvider();

  const alert = useAlertDialog();
  const ask = useAskDialog();
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
    const dropHint = isWorkspaceSupported()
      ? `Drop a single project file (${getProjectFileExtensions().join(", ")}) or workspace folder.`
      : `Drop a single project file (${getProjectFileExtensions().join(", ")}).`;
    if (item === undefined || items.length > 1) {
      await alert({
        title: "Cannot open the dropped items",
        body: dropHint,
      });
      return;
    }
    if (item.handle?.kind === "directory") {
      const directory = item.handle as FileSystemDirectoryHandle;
      if (workspace !== null && (await workspace.isSameEntry(directory))) {
        await alert({
          title: "Workspace already open",
          body: `The folder "${directory.name}" is already the workspace.`,
        });
        return;
      }
      const confirmed = await confirm({
        title: "Open workspace",
        body:
          workspace !== null
            ? `Open the folder "${directory.name}" as the workspace instead of "${workspace.name}"?`
            : `Open the folder "${directory.name}" as the workspace?`,
        actionLabel: "Open",
      });
      if (confirmed) {
        setWorkspace(directory);
      }
      return;
    }
    const projectFile =
      item.handle?.kind === "file"
        ? (item.handle as FileSystemFileHandle)
        : item.file;
    if (projectFile === null) {
      await alert({
        title: "Cannot open the dropped item",
        body: `The dropped item cannot be read. ${dropHint}`,
      });
      return;
    }
    const outsideWorkspaceNote =
      workspace !== null &&
      (projectFile instanceof File ||
        (await resolveProjectSource(projectFile, workspace)) === null)
        ? `"${projectFile.name}" is not in the workspace "${workspace.name}". Its relative paths will be looked up in "${workspace.name}" instead of next to the project file, so some of its data may fail to load.`
        : undefined;
    const foundProjectProviders = findLocalProjectProviders(projectFile);
    let action: string | null;
    let projectProvider: ProjectProvider | null;
    if (isProjectOpen) {
      const question = `Import "${projectFile.name}" into the open project, or open it instead? Opening it replaces the open project, and unsaved changes are lost.`;
      action = await ask({
        title: "Open or import project",
        body:
          outsideWorkspaceNote !== undefined
            ? `${outsideWorkspaceNote} ${question}`
            : question,
        actions: [
          { value: "import", label: "Import" },
          { value: "open", label: "Open" },
        ],
      });
      if (action === null) {
        return;
      }
      projectProvider = await chooseProjectProvider(
        projectFile.name,
        foundProjectProviders,
        { actionLabel: action === "import" ? "Import" : "Open" },
      );
    } else if (foundProjectProviders.length === 1) {
      action = "open";
      const question = `Open the project "${projectFile.name}"?`;
      const confirmed = await confirm({
        title: "Open project",
        body:
          outsideWorkspaceNote !== undefined
            ? `${outsideWorkspaceNote} ${question}`
            : question,
        actionLabel: "Open",
      });
      projectProvider = confirmed ? (foundProjectProviders[0] ?? null) : null;
    } else {
      // choosing the format confirms the drop, too
      action = "open";
      projectProvider = await chooseProjectProvider(
        projectFile.name,
        foundProjectProviders,
        { note: outsideWorkspaceNote, actionLabel: "Open" },
      );
    }
    if (projectProvider === null) {
      return;
    }
    try {
      if (action === "open") {
        await loadProjectFile(projectFile, projectProvider);
      } else if (projectFile instanceof File) {
        await importProjectFromFile(projectProvider, projectFile);
      } else {
        await importProjectFromFileHandle(
          projectProvider,
          projectFile,
          workspace,
        );
      }
    } catch (error) {
      console.error(`Failed to ${action} the dropped project`, error);
      await alert({
        title:
          action === "open"
            ? "Cannot open the project"
            : "Cannot import the project",
        body: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const getDataDrop = <TDataSource extends DataSource>(
    params: AddDataObjectDialogParams<TDataSource>,
  ) => {
    if (
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
            body: `Only files and folders within the workspace can be added. These dropped items are skipped: ${skippedNames.join(", ")}`,
          });
        }
        addDataObject(params, sources);
      },
    };
  };

  switch (panelId) {
    case PanelId.project:
      return {
        label: isProjectOpen ? "Drop to open or import" : "Drop to open",
        accepts: [
          ...(isWorkspaceSupported() ? ["Workspace"] : []),
          `Project file (${getProjectFileExtensions().join(", ")})`,
        ],
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
