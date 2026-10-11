# Changesets

This folder is managed by [changesets](https://github.com/changesets/changesets). Every pull request that changes a published `@tissuumaps/*` package or the `tissuumaps` app adds a changeset:

```sh
pnpm changeset
```

Keep each changeset to one or two sentences on what changed, plus a `Breaking:` line if needed; details belong in the pull request. If a change affects several packages for different readers (e.g. a feature in the app and a new API in a package), add one changeset per package.

Changes that need no release (e.g. tests only) add an empty changeset (`pnpm changeset --empty`); continuous integration fails for changed packages without a changeset.

`pnpm changeset version` turns the pending changesets into version bumps and changelog entries, and `pnpm run release` publishes the packages. See the [development workflow](../apps/docs/docs/development/development-workflow.md#versioning-and-changelogs) for details.
