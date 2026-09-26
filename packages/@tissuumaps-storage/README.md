# @tissuumaps/storage

## Known issues

### geotiff needs to be patched

geotiff 3.0.5 misreads deferred tag values of big-endian TIFFs, and corrupts
LZW-compressed tiles that it decodes on the main thread. The fixes are merged
upstream but not yet released. Until they are, apply
[`geotiff@3.0.5.patch`](https://github.com/TissUUmaps/TissUUmaps4/blob/8a972b9de0b86f0ace08ca41541a746ee2405022/patches/geotiff@3.0.5.patch)
to geotiff 3.0.5 in your project. With pnpm, save it as
`patches/geotiff@3.0.5.patch` and register it in `pnpm-workspace.yaml`:

```yaml
patchedDependencies:
  geotiff@3.0.5: patches/geotiff@3.0.5.patch
```

The patch is a git diff relative to the geotiff package root, so it can also
be applied with other package managers' patching mechanisms. See
[Dependencies](https://tissuumaps.github.io/TissUUmaps4/docs/docs/development/dependencies/)
for details.
