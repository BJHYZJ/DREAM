# Controller source archive

`selected_profiles_v1.zip` contains the Python controllers, execution helpers, task configurations, and associated records used by the simulation profiles. The adjacent lock lists the archive checksum and each member's size and SHA256.

The `dream_sim` package verifies and extracts the archive automatically. To locate an earlier archived demonstration profile (these IDs are separate from the current 50 trial numbers):

```bash
python -m dream_sim.sources --case 01
```

The extracted workspace keeps the relative paths required by the controller imports and resource loaders. See the [architecture guide](../../docs/architecture.md) for the module map and source-cache layout, and the [run guide](../../docs/reproduction.md) for environment setup.

For the current residential controller, the loader applies `controllers/continuous_return/` and its declared parent overrides. `python -m dream_sim.evaluate` verifies the reconstructed source against the current 38/50 archive.
