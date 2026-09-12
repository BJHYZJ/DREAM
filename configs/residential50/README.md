# Residential pick-and-place tasks

Fifty distinct residential scenes, one task per scene, with dynamic memory and seed 42. Each of the five object/receptacle combinations appears in ten scenes.

`task_manifest.json` lists every task and binds its configuration and room map to SHA-256 checksums. `assets.lock.json` records the required AI2-THOR assets at a pinned source revision. `render_assets.lock.json` records the cache used for rendering, including a texture-coordinate index repair for one upstream lamp mesh. The setup guide prepares this cache while preserving the original files.

Room geometry is used to construct the environment and evaluate outcomes; it is not provided to the acting policy.

See the [run guide](../../docs/reproduction.md) for asset preparation and execution.
