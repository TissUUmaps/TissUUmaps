/** The picker APIs missing from `lib.dom.d.ts` */
type FileSystemAccessWindow = Window & {
  showDirectoryPicker?: (options?: {
    id?: string;
    mode?: "read" | "readwrite";
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
export async function pickWorkspace(): Promise<FileSystemDirectoryHandle | null> {
  const w = window as FileSystemAccessWindow;
  if (w.showDirectoryPicker === undefined) {
    throw new Error("Picking a directory is not supported by this browser");
  }
  try {
    // Called as a method: the picker throws if it loses its receiver
    return await w.showDirectoryPicker({ id: workspacePickerId, mode: "read" });
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
  const w = window as FileSystemAccessWindow;
  if (w.showOpenFilePicker === undefined) {
    throw new Error("Picking a file is not supported by this browser");
  }
  try {
    // Called as a method: the picker throws if it loses its receiver
    const projectFiles = await w.showOpenFilePicker({
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
 * Returns whether an error is the one a picker throws when the user cancels it
 *
 * @param error - The error thrown by a picker
 * @returns `true` for an `AbortError`, `false` for any other error
 */
function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
