---
sidebar_position: 4
---

# Development workflow

## Editing

The repository includes a list of recommended extensions for VSCode (`.vscode/extensions.json`) and sample workspace settings (`.vscode/settings.sample.json`, to be copied to `settings.json`).

## Running and building

```sh
pnpm run dev        # application development server
pnpm run docs       # documentation development server
pnpm run build      # packages and application (excludes the documentation)
pnpm run build:docs # packages and documentation
```

Building the documentation for deployment takes the environment variables described in [Code architecture](./code-architecture.md#documentation-docs).

Type checking is part of each package's `build` (`tsc -b`); `pnpm run typecheck` type-checks all projects (emitting only the packages' declaration files into `build/`), including the configuration files and the release scripts.

## Linting

The code can be linted using ESLint:

```sh
pnpm run lint
pnpm run lint:fix
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

The command asks which packages are affected and whether the change is a `patch` (bug fix), `minor` (new, backwards-compatible functionality) or `major` (breaking change) bump, and writes a markdown file to `.changeset/`, which is committed along with the change. Pull requests that touch neither a package nor the application (documentation, tooling, tests) need no changeset.

`pnpm changeset version` consumes the pending changesets: it bumps the affected versions, updates the version ranges between the packages, and prepends the entries to each package's `CHANGELOG.md`. The changelogs are generated; do not edit them by hand. `pnpm run release` builds the packages and publishes those whose version is not on npm yet (`changeset publish`).

The project is currently in changesets' pre-release mode (`.changeset/pre.json`, tag `beta`): versions are computed as usual and suffixed with `-beta.N`, so the first releases are `0.1.0-beta.0` for the packages and `4.0.0-beta.0` for the application. `pnpm changeset pre exit` leaves pre-release mode, after which the next version bump produces the stable versions.

## Pre-commit hooks

Pre-commit hooks for linting and formatting are managed using Husky and lint-staged.

Pre-commit hooks are automatically installed during `pnpm install` using the `prepare` script.

## Continuous integration

Continuous integration is powered by GitHub Actions, for pushes to `main` and for pull requests into it.

Linting, formatting, type checking and testing (see above) need to pass without errors before merging a pull request. Formatting, linting and type checking are checked on pull requests only; the tests, including those of the release scripts, run on every push and pull request.

A review is automatically requested from Copilot and needs to be resolved for every pull request.

Test coverage is [reported to codecov.io](https://app.codecov.io/gh/TissUUmaps/TissUUmaps) for every push and for every pull request.

## Continuous delivery

Releases are automated with changesets and GitHub Actions (`.github/workflows/release.yaml`), on every push to `main`:

1. While changesets are pending, the workflow opens or updates a "Version Packages" pull request that applies them (see [Versioning and changelogs](#versioning-and-changelogs)).
2. Merging that pull request publishes the bumped packages to npm (`pnpm run release`), tags the releases (`@tissuumaps/core@0.1.0-beta.0`, `tissuumaps@4.0.0-beta.0`, ...) and creates the corresponding GitHub releases with the changelog entries as notes.
3. If the application was released, the workflow builds its site from the _published_ packages, never from the workspace sources (see [Release scripts](./code-architecture.md#release-scripts-scripts)): the single-file application with its public files, uploaded to the GitHub release as `tissuumaps-<version>.zip`, which can be hosted on any web server; and its documentation as `tissuumaps-<version>-docs.zip`, built for its deployed path and only used by the deployment.
4. The GitHub Pages site is re-assembled and deployed (see below).

Publishing needs the `NPM_TOKEN` repository secret until the packages exist on npm and trusted publishing is configured for them. A manual run of the workflow (`workflow_dispatch`) rebuilds the site asset of the given release tag, or, with the tag left empty, just re-assembles and deploys the site. The "Version Packages" pull request is opened with the workflow's own token, which does not trigger the continuous integration checks; close and reopen it to run them before merging.

## Continuous deployment

The application and its documentation are deployed to GitHub Pages by version, assembled from the release assets (see [Release scripts](./code-architecture.md#release-scripts-scripts)):

- https://tissuumaps.github.io/TissUUmaps/ redirects to the latest version, https://tissuumaps.github.io/TissUUmaps/docs/ to its documentation.
- `https://tissuumaps.github.io/TissUUmaps/<version>/` is the application of a version, with its documentation under `docs/`.
- Of every MAJOR.MINOR line only the newest version is kept (plus a prerelease above it, if any); every other version redirects to the version that replaces it, keeping the rest of the path (the exact rules are described in [Code architecture](./code-architecture.md#release-scripts-scripts)).
- While no version can be deployed (none has been released, or no release has its assets yet), the current `main` is deployed instead: the application at the root and its documentation under `docs/`.

The Matomo snippet in `.github/pages/custom.html` is inserted into the application page of every deployed version at deployment; the release assets stay free of it. The site also mirrors the plugin index of TissUUmaps 3 under `plugins/` from the repository named by the `V3_PLUGINS_REPO` repository variable, since TissUUmaps 3 loads its plugins from this location.
