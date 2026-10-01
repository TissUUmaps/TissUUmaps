# Changesets

This folder is managed by [changesets](https://github.com/changesets/changesets). Every pull request that changes a published `@tissuumaps/*` package or the `tissuumaps` app adds a changeset:

```sh
pnpm changeset
```

Changes that need no release (e.g. tests only) add an empty changeset (`pnpm changeset --empty`); continuous integration fails for changed packages without a changeset.

`pnpm changeset version` turns the pending changesets into version bumps and changelog entries, and `pnpm run release` publishes the packages. See the [development workflow](../apps/docs/docs/development/development-workflow.md#versioning-and-changelogs) for details.
