import { SourceUtils } from "@tissuumaps/core";

declare global {
  /** The picker APIs missing from `lib.dom.d.ts` */
  interface Window {
    showDirectoryPicker?: (options?: {
      id?: string;
      mode?: "read" | "readwrite";
      startIn?: FileSystemHandle;
    }) => Promise<FileSystemDirectoryHandle>;
    showSaveFilePicker?: (options?: {
      id?: string;
      startIn?: FileSystemHandle;
      suggestedName?: string;
      types?: { description?: string; accept: Record<string, string[]> }[];
    }) => Promise<FileSystemFileHandle>;
    showOpenFilePicker?: (options?: {
      id?: string;
      startIn?: FileSystemHandle;
      multiple?: boolean;
      types?: { description?: string; accept: Record<string, string[]> }[];
    }) => Promise<FileSystemFileHandle[]>;
  }
}

/**
 * Identifies the picker, so that the browser reopens it in the directory it
 * was last used in
 */
const workspacePickerId = "tissuumaps-workspace";

/** The file extensions of project files */
export const projectFileExtensions = [".tm4"];

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
 * @param options - Optional directory to open the picker in, instead of the
 * one it was last used in
 * @returns The file handle, or `null` if the user cancelled the picker
 * @throws Error if the browser does not support picking a file, or if access
 * to the file was denied
 */
export async function pickProjectFile(options?: {
  startIn?: FileSystemDirectoryHandle;
}): Promise<FileSystemFileHandle | null> {
  if (window.showOpenFilePicker === undefined) {
    throw new Error("Picking a file is not supported by this browser");
  }
  try {
    // Called as a method: the picker throws if it loses its receiver
    const projectFiles = await window.showOpenFilePicker({
      id: workspacePickerId,
      startIn: options?.startIn,
      multiple: false,
      types: [
        {
          description: "TissUUmaps project",
          accept: { "application/json": projectFileExtensions },
        },
      ],
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
  const segments = await workspace.resolve(handle);
  if (segments === null) {
    throw new Error(
      `The ${kind === "file" ? "file" : "folder"} is not in the connected folder`,
    );
  }
  if (segments.length === 0) {
    throw new Error(
      "The connected folder itself cannot be a data source; pick a folder inside it",
    );
  }
  return SourceUtils.makeWorkspacePath(segments);
}

/**
 * Lets the user choose where to save a project file within the workspace
 *
 * @param workspace - The directory handle of the open workspace, which the
 * picker opens in
 * @param suggestedName - The file name the picker suggests
 * @returns The handle of the chosen file and its workspace-relative path (with
 * `/` prefix), or `null` if the user cancelled the picker
 * @throws Error if the browser does not support saving a file, if access to
 * the file was denied, or if the file does not lie within the workspace
 */
export async function pickWorkspaceSaveFile(
  workspace: FileSystemDirectoryHandle,
  suggestedName: string,
): Promise<{ file: FileSystemFileHandle; source: string } | null> {
  if (window.showSaveFilePicker === undefined) {
    throw new Error("Saving a file is not supported by this browser");
  }
  let file: FileSystemFileHandle;
  try {
    // Called as a method: the picker throws if it loses its receiver
    file = await window.showSaveFilePicker({
      id: workspacePickerId,
      startIn: workspace,
      suggestedName,
      types: [
        {
          description: "TissUUmaps project",
          accept: { "application/json": projectFileExtensions },
        },
      ],
    });
  } catch (error) {
    if (isAbortError(error)) {
      return null;
    }
    throw error;
  }
  const source = await resolveWorkspacePath(workspace, file);
  if (source === null) {
    throw new Error("The file is not in the connected folder");
  }
  return { file, source };
}

/**
 * Locates a file or directory within the workspace
 *
 * @param workspace - The directory handle of the open workspace
 * @param handle - The file or directory to locate
 * @returns The workspace-relative path of the file or directory (with `/`
 * prefix), or `null` if it does not lie within the workspace
 */
export async function resolveWorkspacePath(
  workspace: FileSystemDirectoryHandle,
  handle: FileSystemHandle,
): Promise<string | null> {
  const segments = await workspace.resolve(handle);
  return segments !== null ? SourceUtils.makeWorkspacePath(segments) : null;
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
