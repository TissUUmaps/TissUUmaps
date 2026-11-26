import { SourceUtils } from "@tissuumaps/core";

declare global {
  /** The picker APIs missing from `lib.dom.d.ts` */
  interface Window {
    showDirectoryPicker?: (options?: {
      id?: string;
      mode?: "read" | "readwrite";
      startIn?: FileSystemHandle;
    }) => Promise<FileSystemDirectoryHandle>;
    showOpenFilePicker?: (options?: {
      id?: string;
      startIn?: FileSystemHandle;
      multiple?: boolean;
      types?: { description?: string; accept: Record<string, string[]> }[];
    }) => Promise<FileSystemFileHandle[]>;
    showSaveFilePicker?: (options?: {
      id?: string;
      startIn?: FileSystemHandle;
      suggestedName?: string;
      types?: { description?: string; accept: Record<string, string[]> }[];
    }) => Promise<FileSystemFileHandle>;
  }

  /** The permission API missing from `lib.dom.d.ts` */
  interface FileSystemHandle {
    requestPermission(descriptor?: {
      mode?: "read" | "readwrite";
    }): Promise<PermissionState>;
  }

  /** The drag and drop API missing from `lib.dom.d.ts` */
  interface DataTransferItem {
    getAsFileSystemHandle?: () => Promise<FileSystemHandle | null>;
  }
}

/**
 * Identifies the picker, so that the browser reopens it in the directory it
 * was last used in
 */
const workspacePickerId = "tissuumaps-workspace";

/**
 * Returns whether the browser can pick a workspace directory
 *
 * @returns `true` if {@link pickWorkspace} is available
 */
export function isWorkspaceSupported(): boolean {
  return typeof window.showDirectoryPicker === "function";
}

/**
 * Lets the user pick the workspace directory
 *
 * The directory is opened for reading only.
 *
 * @returns The directory handle, or `null` if the user cancelled the picker
 * @throws Error if the browser does not support picking a directory, or if
 * access to the directory was denied
 */
export async function pickWorkspace(): Promise<FileSystemDirectoryHandle | null> {
  if (window.showDirectoryPicker === undefined) {
    throw new Error("Picking a directory is not supported by this browser");
  }
  try {
    // Called as a method: the picker throws if it loses its receiver
    return await window.showDirectoryPicker({
      id: workspacePickerId,
      mode: "read",
    });
  } catch (error) {
    if (isAbortError(error)) {
      return null;
    }
    throw error;
  }
}

/**
 * Lets the user pick a project file
 *
 * @param fileTypes - The types of project files to offer, as filters in the
 * order given: a description, and the file extensions by media type
 * @param options - Optional directory to open the picker in, instead of the
 * one it was last used in
 * @returns The file handle, or `null` if the user cancelled the picker
 * @throws Error if the browser does not support picking a file, or if access
 * to the file was denied
 */
export async function pickProjectFile(
  fileTypes: { description: string; accept: Record<string, string[]> }[],
  options?: { startIn?: FileSystemDirectoryHandle },
): Promise<FileSystemFileHandle | null> {
  if (window.showOpenFilePicker === undefined) {
    throw new Error("Picking a file is not supported by this browser");
  }
  try {
    // Called as a method: the picker throws if it loses its receiver
    const projectFiles = await window.showOpenFilePicker({
      id: workspacePickerId,
      startIn: options?.startIn,
      multiple: false,
      types: fileTypes,
    });
    return projectFiles[0] ?? null;
  } catch (error) {
    if (isAbortError(error)) {
      return null;
    }
    throw error;
  }
}

/**
 * Lets the user pick a new project file, starting in the workspace
 *
 * The browser creates the picked file, or empties it if it exists, as soon as
 * it is picked, so a file outside the workspace is returned as well.
 *
 * @param workspace - The directory handle of the open workspace, which the
 * picker opens in
 * @param suggestedName - The file name to suggest
 * @param fileTypes - The types of project files to offer, as filters in the
 * order given (see {@link pickProjectFile})
 * @returns The file handle and its workspace-relative path (with `/` prefix),
 * or `null` as the path if it does not lie within the workspace; or `null` if
 * the user cancelled the picker
 * @throws Error if the browser does not support saving a file, or if access to
 * the file was denied
 */
export async function pickProjectSaveFile(
  workspace: FileSystemDirectoryHandle,
  suggestedName: string,
  fileTypes: { description: string; accept: Record<string, string[]> }[],
): Promise<{ handle: FileSystemFileHandle; source: string | null } | null> {
  if (window.showSaveFilePicker === undefined) {
    throw new Error("Saving a file is not supported by this browser");
  }
  let handle: FileSystemFileHandle;
  try {
    // Called as a method: the picker throws if it loses its receiver. No
    // picker id, so that the workspace picker keeps its own last directory.
    handle = await window.showSaveFilePicker({
      startIn: workspace,
      suggestedName,
      types: fileTypes,
    });
  } catch (error) {
    if (isAbortError(error)) {
      return null;
    }
    throw error;
  }
  const segments = await workspace.resolve(handle);
  return {
    handle,
    source: segments !== null ? SourceUtils.makeWorkspacePath(segments) : null,
  };
}

/**
 * Lets the user pick a file or directory within the workspace
 *
 * @param workspace - The directory handle of the open workspace, which the
 * picker opens in
 * @param kind - Whether to pick a file or a directory
 * @returns The workspace-relative path of the picked file or directory (with
 * `/` prefix), or `null` if the user cancelled the picker
 * @throws Error if the browser does not support picking a file or directory,
 * if access to it was denied, or if it is the workspace itself or does not lie
 * within the workspace
 */
export async function pickWorkspacePath(
  workspace: FileSystemDirectoryHandle,
  kind: FileSystemHandleKind,
): Promise<string | null> {
  if (
    window.showOpenFilePicker === undefined ||
    window.showDirectoryPicker === undefined
  ) {
    throw new Error(
      "Picking a file or directory is not supported by this browser",
    );
  }
  let handle: FileSystemHandle | undefined;
  try {
    // Called as methods: the pickers throw if they lose their receiver. No
    // picker id, so that the workspace picker keeps its own last directory.
    handle =
      kind === "file"
        ? (await window.showOpenFilePicker({ startIn: workspace }))[0]
        : await window.showDirectoryPicker({
            mode: "read",
            startIn: workspace,
          });
  } catch (error) {
    if (isAbortError(error)) {
      return null;
    }
    throw error;
  }
  if (handle === undefined) {
    return null;
  }
  return await resolveWorkspacePath(handle, workspace);
}

/**
 * Locates a file or directory within the workspace
 *
 * @param handle - The handle of the file or directory
 * @param workspace - The directory handle of the open workspace
 * @returns The workspace-relative path of the file or directory (with `/`
 * prefix)
 * @throws Error if it is the workspace itself or does not lie within the
 * workspace
 */
export async function resolveWorkspacePath(
  handle: FileSystemHandle,
  workspace: FileSystemDirectoryHandle,
): Promise<string> {
  const segments = await workspace.resolve(handle);
  if (segments === null) {
    throw new Error(
      `The ${handle.kind === "file" ? "file" : "folder"} is not in the workspace`,
    );
  }
  if (segments.length === 0) {
    throw new Error(
      "The workspace itself cannot be a data source; pick a folder inside it",
    );
  }
  return SourceUtils.makeWorkspacePath(segments);
}

/**
 * Returns whether an error is the one a picker throws when the user cancels it
 *
 * @param error - The error thrown by a picker
 * @returns `true` for an `AbortError`, `false` for any other error
 */
function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

/**
 * Reads the files and directories dropped in a drop event
 *
 * The items have to be read within the event, before the browser clears its
 * data transfer; their handles are resolved afterwards. Handles are only
 * available in browsers that support them (see {@link isWorkspaceSupported}).
 *
 * @param dataTransfer - The data transfer of the drop event
 * @returns The dropped items, each as its handle (if available) and its file;
 * without a handle, a directory cannot be told apart from a file
 */
export async function readDroppedItems(dataTransfer: DataTransfer): Promise<
  {
    handle: FileSystemHandle | null;
    file: File | null;
  }[]
> {
  const items = Array.from(dataTransfer.items)
    .filter((item) => item.kind === "file")
    .map((item) => ({
      handlePromise: item.getAsFileSystemHandle?.(),
      file: item.getAsFile(),
    }));
  return await Promise.all(
    items.map(async ({ handlePromise, file }) => {
      let handle: FileSystemHandle | null = null;
      if (handlePromise !== undefined) {
        try {
          handle = await handlePromise;
        } catch {
          // an item whose handle cannot be obtained is read without one
        }
      }
      return { handle, file };
    }),
  );
}
