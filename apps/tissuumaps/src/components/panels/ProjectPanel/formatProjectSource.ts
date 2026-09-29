import { SourceUtils } from "@tissuumaps/core";

/**
 * Formats a project source for display
 *
 * Workspace files are shown relative to the folder, and URLs under the app
 * relative to the app.
 *
 * @param source - The project source
 * @param workspaceName - The name of the open workspace, if any
 * @param baseUrl - The base URL of the app, usually `document.baseURI`
 * @returns The displayed source
 */
export function formatProjectSource(
  source: string,
  workspaceName: string | null,
  baseUrl: string,
): string {
  if (SourceUtils.isWorkspacePath(source)) {
    return `${workspaceName ?? "Folder"} › ${source.slice(1)}`;
  }
  if (!URL.canParse(source)) {
    return source;
  }
  const url = new URL(source);
  const appUrl = new URL(".", baseUrl);
  const path = `${url.pathname}${url.search}`;
  return url.origin === appUrl.origin &&
    url.pathname.startsWith(appUrl.pathname)
    ? path.slice(appUrl.pathname.length)
    : `${url.host}${path}`;
}
