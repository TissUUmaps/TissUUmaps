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

Type checking is part of each package's `build` (`tsc -b`); there is no separate type-check script.

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
pnpm run test
```

This is an academic project. As such, we encourage rigorous testing, but loosely tested code may also be acceptable. Which units are expected to have tests is described in [Coding conventions](./coding-conventions.md#tests).

## Documentation

User and developer documentation is written in markdown, rendered using Docusaurus, and deployed to GitHub Pages (see below).

API documentation for packages can be auto-generated from TSDoc code comments.

## Issues and pull requests

Bug reports and feature requests use the issue templates in `.github/ISSUE_TEMPLATE`. Pull requests follow `.github/PULL_REQUEST_TEMPLATE.md`, which includes a declaration of AI use (see [AI policy](./ai-policy.md)).

## Version control

This project uses [semantic versioning](https://semver.org); see [Versioning and changelogs](#versioning-and-changelogs) below.

GitHub is used for distributed version control using Git: https://github.com/TissUUmaps/TissUUmaps4

The repository follows a simplified Git Flow-like branching model, with a `main` branch holding the latest stable version and a single `development` branch, into which feature branches are merged. Branch rules protect both the `main` branch and the `development` branch from direct pushes without pull requests. Commit messages follow the [conventional commits](https://www.conventionalcommits.org) specification, with a scope where one applies and `!` marking breaking changes (e.g. `feat(storage)!: resolve relative URLs against the project URL`); branch names and pull requests should loosely follow [conventional branch](https://conventional-branch.github.io) guidelines. Only signed commits can be merged.

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

Continuous integration is powered by GitHub Actions. Only the `main` and `development` branches are considered.

Linting, formatting, and testing (see above) need to pass without errors before merging a pull request. Formatting and linting are checked on pull requests only; tests run on every push and pull request. The packages are built before linting and testing, which also type-checks them.

A review is automatically requested from Copilot and needs to be resolved for every pull request.

Test coverage is [reported to codecov.io](https://app.codecov.io/gh/TissUUmaps/TissUUmaps4) for every push and for every pull request.

## Continuous deployment

The application and documentation is continuously deployed to GitHub Pages using GitHub Actions.

Current stable version &rarr; `main` branch:

- Application: https://tissuumaps.github.io/TissUUmaps4/live/
- Documentation: https://tissuumaps.github.io/TissUUmaps4/docs/

Current development version &rarr; `development` branch:

- Application: https://tissuumaps.github.io/TissUUmaps4/live-dev/
- Documentation: https://tissuumaps.github.io/TissUUmaps4/docs-dev/

## Continuous delivery

Packages are published to npm from the versions and changelogs produced by changesets (see [Versioning and changelogs](#versioning-and-changelogs)): `pnpm run release` builds the packages and runs `changeset publish`. Publishing is not automated yet.
