import { SourceUtils } from "@tissuumaps/core";

/** The picker APIs missing from `lib.dom.d.ts` */
type FileSystemAccessWindow = Window & {
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
};

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
  return (
    typeof (window as FileSystemAccessWindow).showDirectoryPicker === "function"
  );
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
export function pickWorkspace(): Promise<FileSystemDirectoryHandle | null> {
  return openDirectoryPicker({ id: workspacePickerId });
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
export function pickProjectFile(options?: {
  startIn?: FileSystemDirectoryHandle;
}): Promise<FileSystemFileHandle | null> {
  return openFilePicker({
    startIn: options?.startIn,
    types: [
      {
        description: "TissUUmaps project",
        accept: { "application/json": projectFileExtensions },
      },
    ],
  });
}

/**
 * Lets the user pick a file within the workspace
 *
 * @param workspace - The directory handle of the open workspace, which the
 * picker opens in
 * @returns The workspace-relative path of the picked file (with `/` prefix),
 * or `null` if the user cancelled the picker
 * @throws Error if the browser does not support picking a file, if access to
 * the file was denied, or if the file does not lie within the workspace
 */
export async function pickWorkspaceFilePath(
  workspace: FileSystemDirectoryHandle,
): Promise<string | null> {
  const file = await openFilePicker({ startIn: workspace });
  return file !== null ? await locateInWorkspace(workspace, file) : null;
}

/**
 * Lets the user pick a directory within the workspace
 *
 * @param workspace - The directory handle of the open workspace, which the
 * picker opens in
 * @returns The workspace-relative path of the picked directory (with `/`
 * prefix), or `null` if the user cancelled the picker
 * @throws Error if the browser does not support picking a directory, if access
 * to the directory was denied, or if the directory is the workspace itself or
 * does not lie within it
 */
export async function pickWorkspaceDirectoryPath(
  workspace: FileSystemDirectoryHandle,
): Promise<string | null> {
  // No picker id: the workspace picker keeps its own last directory
  const directory = await openDirectoryPicker({ startIn: workspace });
  return directory !== null
    ? await locateInWorkspace(workspace, directory)
    : null;
}

/**
 * Locates a picked file or directory within the workspace
 *
 * @param workspace - The directory handle of the open workspace
 * @param handle - The picked file or directory
 * @returns The workspace-relative path of the file or directory (with `/`
 * prefix)
 * @throws Error if the file or directory does not lie within the workspace, or
 * is the workspace itself, which is no source
 */
async function locateInWorkspace(
  workspace: FileSystemDirectoryHandle,
  handle: FileSystemHandle,
): Promise<string> {
  const segments = await workspace.resolve(handle);
  if (segments === null) {
    throw new Error(
      `The ${handle.kind === "directory" ? "folder" : "file"} is not in the connected folder`,
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
 * Opens the browser's directory picker, for reading only
 *
 * @param options - The picker id, whose last directory the browser reopens,
 * and the directory to open the picker in instead
 * @returns The directory handle, or `null` if the user cancelled the picker
 * @throws Error if the browser does not support picking a directory, or if
 * access to the directory was denied
 */
async function openDirectoryPicker(options: {
  id?: string;
  startIn?: FileSystemDirectoryHandle;
}): Promise<FileSystemDirectoryHandle | null> {
  const w = window as FileSystemAccessWindow;
  if (w.showDirectoryPicker === undefined) {
    throw new Error("Picking a directory is not supported by this browser");
  }
  try {
    // Called as a method: the picker throws if it loses its receiver
    return await w.showDirectoryPicker({
      id: options.id,
      mode: "read",
      startIn: options.startIn,
    });
  } catch (error) {
    if (isAbortError(error)) {
      return null;
    }
    throw error;
  }
}

/**
 * Opens the browser's file picker for a single file
 *
 * @param options - The directory to open the picker in, and the file types to
 * offer, defaulting to all
 * @returns The file handle, or `null` if the user cancelled the picker
 * @throws Error if the browser does not support picking a file, or if access
 * to the file was denied
 */
async function openFilePicker(options: {
  startIn?: FileSystemDirectoryHandle;
  types?: { description?: string; accept: Record<string, string[]> }[];
}): Promise<FileSystemFileHandle | null> {
  const w = window as FileSystemAccessWindow;
  if (w.showOpenFilePicker === undefined) {
    throw new Error("Picking a file is not supported by this browser");
  }
  try {
    // Called as a method: the picker throws if it loses its receiver
    const files = await w.showOpenFilePicker({
      id: workspacePickerId,
      startIn: options.startIn,
      multiple: false,
      types: options.types,
    });
    return files[0] ?? null;
  } catch (error) {
    if (isAbortError(error)) {
      return null;
    }
    throw error;
  }
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
