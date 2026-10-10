---
sidebar_position: 4
---

# Development workflow

## Editing

The repository includes a list of recommended extensions for VSCode (`.vscode/extensions.json`) and sample workspace settings (`.vscode/settings.sample.json`, to be copied to `settings.json`).

## Running and building

```sh
pnpm run dev            # application development server
pnpm run build          # packages and application
pnpm run build:packages # packages only
pnpm run build:apidocs  # API documentation in docs/api (requires `pnpm run build:packages` first)
```

The documentation is rendered by the [website](https://github.com/TissUUmaps/website). To preview it, clone the website next to this repository and follow its README.

Building the documentation for deployment takes the environment variables described in [Code architecture](./code-architecture.md#documentation-docs).

Type checking is part of each package's `build` (`tsc -b`); `pnpm run typecheck` type-checks all projects (emitting only the packages' declaration files into `build/`), including the configuration files and the release scripts.

## Linting

The code can be linted using ESLint:

```sh
pnpm run lint
pnpm run lint:fix
```

The links between documentation pages, including links into the generated API documentation, are checked using remark (after `pnpm run build:apidocs`):

```sh
pnpm run lint:docs
```

## Formatting

The code can be formatted using Prettier:

```sh
pnpm run fmt
pnpm run fmt:check
```

## Testing

The code can be tested using Vitest:

```sh
pnpm run test         # packages and application (Vitest)
pnpm run test:scripts # release scripts (Node test runner)
```

This is an academic project. As such, we encourage rigorous testing, but loosely tested code may also be acceptable. Which units are expected to have tests is described in [Coding conventions](./coding-conventions.md#tests).

## Documentation

User and developer documentation is written in markdown, rendered using Docusaurus, and deployed to GitHub Pages (see below).

API documentation for packages can be auto-generated from TSDoc code comments.

## Issues and pull requests

Bug reports and feature requests use the issue templates in `.github/ISSUE_TEMPLATE`. Pull requests follow `.github/PULL_REQUEST_TEMPLATE.md`, which includes a declaration of AI use (see [AI policy](./ai-policy.md)).

By submitting a pull request, you confirm that you have authored its content, that you have the necessary rights to it, and that it may be provided under the project's [MIT license](https://github.com/TissUUmaps/TissUUmaps/blob/main/LICENSE).

## Version control

This project uses [semantic versioning](https://semver.org); see [Versioning and changelogs](#versioning-and-changelogs) below.

GitHub is used for distributed version control using Git: https://github.com/TissUUmaps/TissUUmaps

The repository has a single `main` branch, into which feature branches are merged through pull requests; releases are cut from it with changesets (see [Continuous delivery](#continuous-delivery)). Branch rules protect `main` from direct pushes. Commit messages follow the [conventional commits](https://www.conventionalcommits.org) specification, with a scope where one applies and `!` marking breaking changes (e.g. `feat(storage)!: resolve relative URLs against the project URL`); branch names and pull requests should loosely follow [conventional branch](https://conventional-branch.github.io) guidelines. Only signed commits can be merged.

## Versioning and changelogs

Versions and changelogs are managed with [changesets](https://github.com/changesets/changesets). The published `@tissuumaps/*` packages are versioned independently of each other; the `tissuumaps` application has its own version (4.x). The documentation is not versioned.

Every pull request that changes a published package or the application adds a changeset describing the change from a user's perspective:

```sh
pnpm changeset
```

The command asks which packages are affected and whether the change is a `patch` (bug fix), `minor` (new, backwards-compatible functionality) or `major` (breaking change) bump, and writes a markdown file to `.changeset/`, which is committed along with the change. Pull requests that touch neither a package nor the application (documentation, root tooling) need no changeset. Changes inside a package or the application that need no release (e.g. tests only) add an empty changeset instead (`pnpm changeset --empty`), since continuous integration fails for changed packages without a changeset.

`pnpm changeset version` consumes the pending changesets: it bumps the affected versions, updates the version ranges between the packages, and prepends the entries to each package's `CHANGELOG.md`. The changelogs are generated; do not edit them by hand. `pnpm run release` builds the packages and publishes those whose version is not on npm yet (`changeset publish`).

The project is currently in changesets' pre-release mode (`.changeset/pre.json`, tag `beta`): versions are computed as usual and suffixed with `-beta.N`, so the first releases are `0.1.0-beta.0` for the packages and `4.0.0-beta.0` for the application. `pnpm changeset pre exit` leaves pre-release mode, after which the next version bump produces the stable versions.

## Pre-commit hooks

Pre-commit hooks for linting and formatting are managed using Husky and lint-staged.

Pre-commit hooks are automatically installed during `pnpm install` using the `prepare` script.

## Continuous integration

Continuous integration is powered by GitHub Actions, for pushes to `main` and for pull requests into it.

Linting, formatting, type checking, building and testing (see above) need to pass without errors before merging a pull request:

- On pull requests only: formatting, type checking and linting, the builds of the packages, the application and the documentation (the same builds as for a release), and that a pull request changing a package or the application adds a changeset (`changeset status`, skipped for the "Version Packages" pull request).
- On every push and pull request: the tests, including those of the release script.

A review is automatically requested from Copilot and needs to be resolved for every pull request.

Test coverage is [reported to codecov.io](https://app.codecov.io/gh/TissUUmaps/TissUUmaps) for every push and for every pull request.

## Continuous delivery

Releases are automated with changesets and GitHub Actions (`.github/workflows/release.yaml`), on every push to `main`:

1. While changesets are pending, the workflow opens or updates a "Version Packages" pull request that applies them (see [Versioning and changelogs](#versioning-and-changelogs)).
2. Merging that pull request publishes the bumped packages to npm (`pnpm run release`), tags the releases (`@tissuumaps/core@0.1.0-beta.0`, `tissuumaps@4.0.0-beta.0`, ...) and creates the corresponding GitHub releases with the changelog entries as notes.
3. If the application was released, the workflow builds its site and uploads it to the application's GitHub release:
   - `tissuumaps-<version>.zip`: the single-file application with its public files, which can be hosted on any web server.
   - `tissuumaps-<version>-docs.zip`: its documentation, built for its deployed path and only used by the deployment.

   The site is built from the repository at the application's release tag, without waiting for the packages to appear on npm. As long as every change to a package comes with a changeset for that package, the package sources at that tag are those of the published versions.

4. The GitHub Pages site is re-assembled and deployed (see below).

Further details:

- Packages are published through npm trusted publishing (OIDC), so no npm token is needed.
- The release workflow only runs in the `TissUUmaps/TissUUmaps` repository, never in forks.
- A manual run of the workflow (`workflow_dispatch`) rebuilds the site assets of the given release tag. With the tag left empty, it only re-assembles and deploys the site.
- The "Version Packages" pull request is opened with the workflow's own token, which does not trigger the continuous integration checks. Close and reopen it to run them before merging.

## Continuous deployment

The application and its documentation are deployed to GitHub Pages, one MAJOR.MINOR line at a time, assembled from the release assets (the exact rules are described in [Release scripts](./code-architecture.md#release-scripts-scripts)):

- `https://tissuumaps.github.io/TissUUmaps/<MAJOR.MINOR>/` (e.g. `4.0/`) is the application of a line, with its documentation under `docs/`. It serves the line's latest stable version, or its latest prerelease while the line has no stable version.
- https://tissuumaps.github.io/TissUUmaps/ redirects to the latest line, and https://tissuumaps.github.io/TissUUmaps/docs/ to its documentation.
- Paths of lines that are not deployed (abandoned prerelease lines) redirect to the latest line.
- While no version can be deployed (none has been released, or no release has its assets yet), the deployment fails and the deployed site is left as it is.

Two things are added at deployment:

- The Matomo snippet in `.github/pages/custom.html`, inserted into the application page of every deployed line. The release assets stay free of it.
- A mirror of the TissUUmaps 3 plugin index under `plugins/`, copied from the repository named by the `V3_PLUGINS_REPO` repository variable, since TissUUmaps 3 loads its plugins from this location.
