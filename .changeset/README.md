# Changesets

This folder is managed by [changesets](https://github.com/changesets/changesets). Every pull request that changes a published `@tissuumaps/*` package or the `tissuumaps` app adds a changeset:

```sh
pnpm changeset
```

`pnpm changeset version` turns the pending changesets into version bumps and changelog entries, and `pnpm run release` publishes the packages. See the [development workflow](../apps/docs/docs/development/development-workflow.md#versioning-and-changelogs) for details.
