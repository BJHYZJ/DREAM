# Task configuration

`cases.json` maps Video 01–10 to their task profiles. Each entry specifies the house, seed, instruction, controller version, task inputs, and evaluation settings.

- `tasks/`: task definitions and evaluator room maps.
- `locks/`: model revisions and scene-asset checksums.

The runner validates these files against the catalog in the source archive. Paths in the catalog's `records` and source fields are relative to the extracted catalog. Run `python -m dream_sim.sources --case 01` to locate that source tree.

Task files record the configuration at creation time. Their `constructed_not_executed` field describes that stage; execution outcomes are stored separately in the [experiment records](../reproducibility/evidence/).

Existing profiles are checksum-locked to their recorded controller and inputs. Create a new profile for a different seed, task, or controller version.
