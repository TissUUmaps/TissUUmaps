/**
 * Redirect rules of the versioned GitHub Pages site
 *
 * This module is inlined into the site's redirect pages (root, `404.html`, and
 * the stubs of superseded versions) by `scripts/assemble-pages.ts`, so it must
 * stay free of imports and of syntax that is not plain browser JavaScript once
 * its types are stripped.
 */

/** The site description inlined into the redirect pages */
export type Site = {
  /** The path prefix of the site, e.g. `/TissUUmaps/` */
  prefix: string;
  /** The version the site root redirects to, or `null` while no version is deployed */
  latest: string | null;
  /** The versions that are deployed */
  retained: string[];
  /** Superseded versions mapped to the deployed version that replaces them */
  redirects: Record<string, string>;
};

/**
 * Resolves where a path of the site should redirect to
 *
 * Superseded versions redirect to their replacement keeping the rest of the
 * path; `docs` redirects to the latest documentation; deployed versions never
 * redirect (a missing page there is a genuine 404); everything else, including
 * the site root, redirects to the latest version's root.
 *
 * @param pathname - The requested path, e.g. `window.location.pathname`
 * @param site - The site description
 * @returns The target path, or `null` if the path should not redirect
 */
export function resolveRedirect(pathname: string, site: Site): string | null {
  if (site.latest === null || !pathname.startsWith(site.prefix)) {
    return null;
  }
  const rest = pathname.slice(site.prefix.length);
  const slash = rest.indexOf("/");
  const segment = slash === -1 ? rest : rest.slice(0, slash);
  const tail = slash === -1 ? "/" : rest.slice(slash);
  let target: string;
  if (Object.hasOwn(site.redirects, segment)) {
    target = site.prefix + site.redirects[segment] + tail;
  } else if (segment === "docs") {
    target = site.prefix + site.latest + "/docs" + tail;
  } else if (site.retained.includes(segment)) {
    return null;
  } else {
    target = site.prefix + site.latest + "/";
  }
  return target === pathname ? null : target;
}
