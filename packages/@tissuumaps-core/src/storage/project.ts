import type { Project, RawProject } from "../model/project";
import type { ProgressCallback } from "../types/callbacks";

/**
 * Reads projects in a project file format, writes them, or both
 *
 * The project-level counterpart of `DataProvider`: where a data provider opens
 * the data behind a data object, a project provider opens a whole project,
 * e.g. a TissUUmaps project file or a SpatialData dataset. The application
 * either opens a loaded project, replacing the current one, or imports it,
 * merging it into the current one.
 *
 * Its members serve three roles:
 *
 * - recognizing what it loads: {@link ProjectProvider.fileTypes} for local
 *   files, {@link ProjectProvider.supports} for sources
 * - loading: {@link ProjectProvider.load}, also from the GET parameter
 *   {@link ProjectProvider.urlParam}
 * - writing: {@link ProjectProvider.toFile}
 *
 * `load` and `toFile` are both optional, as some formats can only be read and
 * others only be written, but a project provider has to implement at least
 * one of them. The members for recognizing and loading are only used for
 * project providers that implement `load`.
 *
 * A source can be the workspace root `/`, e.g. for a directory that is about
 * to become the workspace.
 */
export interface ProjectProvider {
  /** The human-readable name of the project provider */
  readonly name: string;

  /**
   * The GET parameter naming a source to load with this project provider, if
   * any (e.g. `spatialdata` for `?spatialdata=…`)
   */
  readonly urlParam?: string;

  /**
   * The project files this project provider loads, as their extensions by
   * media type, in lowercase and with leading dot (e.g.
   * `{ "application/json": [".tm4"] }`)
   *
   * Used to pick a project provider for a local file, matching its name
   * regardless of case, and to filter file pickers. Local files are matched
   * this way also within the workspace, as those outside of it have no source
   * to pass to {@link ProjectProvider.supports}.
   */
  readonly fileTypes?: Record<string, string[]>;

  /**
   * Returns whether this project provider can likely load a source
   *
   * Used to pick a project provider for a source the user entered or dropped,
   * e.g. by its extension or by cheaply reading its metadata. A project
   * provider without this method is never picked for a source automatically,
   * but can still be chosen by the user. Callers treat a rejection as `false`.
   *
   * @param normalizedSource - The source, normalized with
   * `SourceUtils.normalizeSource`
   * @param workspace - The directory handle of the workspace, if any
   * @param options - Optional abort signal
   * @returns A promise that resolves to whether the source is supported
   */
  supports?(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<boolean>;

  /**
   * Loads a project from a source
   *
   * The data sources of the project are returned as authored: relative sources
   * are allowed, and preferred, so that the project stays portable. They are
   * resolved against the returned `source` with ordinary URL semantics, like
   * those of a project file against the file's location, so a dataset that is
   * a directory includes its own name in them (e.g. `data.zarr/images/a`),
   * unless it is the workspace root (e.g. `images/a`).
   *
   * Every ID in the project has to be unique within it, and must not contain
   * `:`. The application rejects merging the project into another one that
   * already uses one of its IDs, so a project provider that creates IDs
   * should make them unique across projects (e.g. with `SourceUtils.createID`).
   *
   * @param normalizedSource - The source, normalized with
   * `SourceUtils.normalizeSource`
   * @param options - Optional abort signal, the directory handle of the
   * workspace, and a progress callback
   * @returns A promise that resolves to the project, and to where it was
   * actually read from: the absolute URL after following any redirects, or the
   * workspace-relative path. The application resolves the project's relative
   * sources against it, as only the project provider knows it.
   */
  load?(
    normalizedSource: string,
    options?: {
      signal?: AbortSignal;
      workspace?: FileSystemDirectoryHandle | null;
      onProgress?: ProgressCallback;
    },
  ): Promise<{ project: RawProject; source: string }>;

  /**
   * Writes a project to a single file, which the application downloads
   *
   * The project has been detached from the project store beforehand, and holds
   * the project's own properties only. Its data sources are written as they
   * are.
   *
   * @param project - The project to write
   * @param options - Optional name of the file without extension, to use as is
   * (e.g. that of the file the project was opened from), abort signal and
   * progress callback
   * @returns A promise that resolves to the file, named as given or else after
   * the project, and with the extension of the format
   */
  toFile?(
    project: Project,
    options?: {
      name?: string;
      signal?: AbortSignal;
      onProgress?: ProgressCallback;
    },
  ): Promise<File>;
}
